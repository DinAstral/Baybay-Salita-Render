require("dotenv").config();
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const fsp = require("fs").promises;
const ffmpeg = require("fluent-ffmpeg");
const wav = require("wav-decoder");
const Meyda = require("meyda");
const CompareModel = require("../models/ComparisonResult");

// Download audio from URL
const downloadAudio = async (url, filename) => {
  const response = await axios({ method: "GET", url, responseType: "stream" });
  return new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(filename);
    response.data.pipe(writer);
    writer.on("finish", resolve);
    writer.on("error", reject);
  });
};

// Delete local file if exists
const deleteFileIfExists = (path) => {
  if (fs.existsSync(path)) fs.unlinkSync(path);
};

// Chunk audio buffer
const chunkArray = (arr, size) => {
  const chunks = [];
  for (let i = 0; i < arr.length; i += size) {
    const chunk = arr.slice(i, i + size);
    if (chunk.length < size) {
      const padded = new Float32Array(size);
      padded.set(chunk);
      chunks.push(padded);
    } else {
      chunks.push(chunk);
    }
  }
  return chunks;
};

// STT using ElevenLabs (Filipino)
const transcribeAudio = async (filePath) => {
  const formData = new FormData();
  formData.append("audio", fs.createReadStream(filePath));
  formData.append("model_id", "eleven_multilingual_v1");

  const headers = {
    ...formData.getHeaders(),
    "xi-api-key": process.env.ELEVENLABS_API_KEY,
  };

  try {
    const response = await axios.post(
      "https://api.elevenlabs.io/v1/speech-to-text",
      formData,
      { headers }
    );
    return response.data.transcript;
  } catch (error) {
    console.error("Error in STT:", error.response?.data || error.message);
    return null;
  }
};

// Voice detection (based on ZCR and energy)
const detectSilentOrLowVoiceAudio = (features) => {
  const zcrThreshold = 10.0;
  const energyThreshold = 0.03;
  const avgZCR = features.zcr.reduce((a, b) => a + b, 0) / features.zcr.length;
  const avgEnergy = features.energy ?? 0;
  return avgZCR < zcrThreshold && avgEnergy < energyThreshold;
};

// Feature extraction using Meyda
const extractAudioFeatures = async (audioPath) => {
  const tempOutput = `temp_${audioPath}`;
  deleteFileIfExists(tempOutput);

  return new Promise((resolve, reject) => {
    ffmpeg(audioPath)
      .toFormat("wav")
      .on("end", async () => {
        try {
          const data = await fsp.readFile(tempOutput);
          const audioData = await wav.decode(data);
          const channelData = audioData.channelData[0];
          const bufferSize = 512;
          const chunks = chunkArray(channelData, bufferSize);
          const totalChunks = chunks.length;

          const sumFeatures = chunks.reduce((acc, chunk, i) => {
            const features = Meyda.extract(["mfcc", "chroma", "zcr"], chunk);
            if (i === 0) {
              for (let key in features) {
                acc[key] = Array.isArray(features[key])
                  ? features[key].slice()
                  : [features[key]];
              }
            } else {
              for (let key in features) {
                if (Array.isArray(features[key])) {
                  acc[key] = acc[key].map((val, idx) => val + features[key][idx]);
                } else {
                  acc[key][0] += features[key];
                }
              }
            }
            return acc;
          }, {});

          for (let key in sumFeatures) {
            sumFeatures[key] = sumFeatures[key].map((val) => val / totalChunks);
          }

          resolve(sumFeatures);
        } catch (err) {
          reject(err);
        }
      })
      .on("error", reject)
      .save(tempOutput);
  });
};

// Compare extracted audio features
const compareAudioFeatures = (f1, f2) => {
  const euclidean = (a, b) =>
    Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
  return {
    mfccDistance: euclidean(f1.mfcc, f2.mfcc),
    chromaDistance: euclidean(f1.chroma, f2.chroma),
    zcr: euclidean(f1.zcr, f2.zcr),
  };
};

// Weighted similarity score
const stentWeightedAudioSimilarity = (mfcc, chroma, zcr) =>
  0.4 * mfcc + 0.3 * chroma + 0.3 * zcr;

// 🎯 Main Comparison
const run = async (defaultAudioUrl, userAudioUrl) => {
  const audio1 = "audio1.wav";
  const audio2 = "audio2.wav";
  const cleanAudio = "clean.wav";

  await downloadAudio(defaultAudioUrl, audio1);
  await downloadAudio(userAudioUrl, audio2);

  // Transcribe to check speech presence
  const transcript = await transcribeAudio(audio2);
  if (!transcript || transcript.trim() === "") {
    console.log("No speech detected in user audio.");
    return {
      audioComparison: null,
      weightedSimilarity: 100,
      remarks: "Incorrect",
    };
  }

  await new Promise((resolve, reject) => {
    ffmpeg(audio2)
      .output(cleanAudio)
      .audioFilters("afftdn=nf=-25")
      .on("end", resolve)
      .on("error", reject)
      .run();
  });

  const features1 = await extractAudioFeatures(audio1);
  const features2 = await extractAudioFeatures(cleanAudio);

  deleteFileIfExists(audio2);
  deleteFileIfExists(cleanAudio);

  if (detectSilentOrLowVoiceAudio(features2)) {
    console.log("Low voice detected.");
    return {
      audioComparison: null,
      weightedSimilarity: 100,
      remarks: "Incorrect",
    };
  }

  const comparison = compareAudioFeatures(features1, features2);
  const similarity = stentWeightedAudioSimilarity(
    comparison.mfccDistance,
    comparison.chromaDistance,
    comparison.zcr
  );

  return {
    audioComparison: comparison,
    weightedSimilarity: similarity,
    remarks: similarity <= 20 ? "Correct" : "Incorrect",
  };
};

// Full comparison loop
const runComparisonAndSaveResult = async (
  UserInputId,
  ActivityCode,
  LRN,
  Section,
  Type,
  fileUrls,
  defaultAudios,
  similarityThreshold = 20
) => {
  const comparisonResults = [];
  let totalScore = 0;

  for (let i = 0; i < defaultAudios.length; i++) {
    const userAudioUrl = fileUrls[`AudioURL${i + 1}`];
    const defaultAudioUrl = defaultAudios[i];
    const result = await run(defaultAudioUrl, userAudioUrl);

    if (result.remarks === "Correct") totalScore++;

    comparisonResults.push({
      ItemCode: `Itemcode${i + 1}`,
      mfccDistance: result.audioComparison?.mfccDistance ?? null,
      chromaDistance: result.audioComparison?.chromaDistance ?? null,
      zcr: result.audioComparison?.zcr ?? null,
      stentWeightedSimilarity: result.weightedSimilarity,
      Remarks: result.remarks,
    });
  }

  await CompareModel.create({
    UserInputId,
    ActivityCode,
    LRN,
    Section,
    Type,
    Results: comparisonResults,
  });

  return { score: totalScore, resultsWithRemarks: comparisonResults };
};

module.exports = {
  runComparisonAndSaveResult,
};
