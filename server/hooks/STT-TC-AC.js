require("dotenv").config();
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const wav = require("wav-decoder");
const Meyda = require("meyda");
const mongoose = require("mongoose");
const CompareModel = require("../models/ComparisonResult");

const downloadAudio = async (url, filename) => {
  const response = await axios({ method: "GET", url, responseType: "stream" });
  return new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(filename);
    response.data.pipe(writer);
    writer.on("finish", resolve);
    writer.on("error", reject);
  });
};

const deleteFileIfExists = (filePath) => {
  if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
};

const chunkArray = (arr, chunkSize) => {
  const chunks = [];
  for (let i = 0; i < arr.length; i += chunkSize) {
    const chunk = arr.slice(i, i + chunkSize);
    const paddedChunk = new Float32Array(chunkSize);
    paddedChunk.set(chunk);
    chunks.push(paddedChunk);
  }
  return chunks;
};

const detectSilentOrLowVoiceAudio = (features) => {
  const zcrThreshold = 10.0;
  const energyThreshold = 0.03;
  const avgZCR = features.zcr.reduce((a, b) => a + b, 0) / features.zcr.length;
  const avgEnergy = features.energy || 0;

  return avgZCR < zcrThreshold && avgEnergy < energyThreshold;
};

const extractAudioFeatures = async (audioPath) => {
  return new Promise((resolve, reject) => {
    const tempOutput = `temp_${audioPath}`;
    deleteFileIfExists(tempOutput);

    ffmpeg(audioPath)
      .toFormat("wav")
      .on("end", () => {
        fs.readFile(tempOutput, async (err, data) => {
          if (err) return reject(err);
          try {
            const audioData = await wav.decode(data);
            const channelData = audioData.channelData[0];
            const bufferSize = 512;
            const audioChunks = chunkArray(channelData, bufferSize);

            const sumFeatures = audioChunks.reduce((acc, chunk, index) => {
              const features = Meyda.extract(["mfcc", "chroma", "zcr"], chunk);
              if (index === 0) {
                Object.keys(features).forEach((key) => {
                  acc[key] = Array.isArray(features[key]) ? [...features[key]] : [features[key]];
                });
              } else {
                Object.keys(features).forEach((key) => {
                  if (Array.isArray(features[key])) {
                    acc[key] = acc[key].map((val, i) => val + features[key][i]);
                  } else {
                    acc[key][0] += features[key];
                  }
                });
              }
              return acc;
            }, {});

            Object.keys(sumFeatures).forEach((key) => {
              sumFeatures[key] = sumFeatures[key].map((val) => val / audioChunks.length);
            });

            resolve(sumFeatures);
          } catch (decodeError) {
            reject(decodeError);
          }
        });
      })
      .on("error", reject)
      .save(tempOutput);
  });
};

const compareAudioFeatures = (features1, features2) => {
  const euclidean = (v1, v2) =>
    Math.sqrt(v1.reduce((acc, val, i) => acc + Math.pow(val - v2[i], 2), 0));

  return {
    mfccDistance: euclidean(features1.mfcc, features2.mfcc),
    chromaDistance: euclidean(features1.chroma, features2.chroma),
    zcr: euclidean(features1.zcr, features2.zcr),
  };
};

const stentWeightedAudioSimilarity = (mfcc, chroma, zcr) => {
  return 0.4 * mfcc + 0.3 * chroma + 0.3 * zcr;
};

// 🔥 ElevenLabs STT Integration
const transcribeAudio = async (filePath) => {
  try {
    const formData = new FormData();
    formData.append("file", fs.createReadStream(filePath)); // ✅ Correct key
    formData.append("model_id", "eleven_multilingual_v1");
    formData.append("language", "fil");

    const response = await axios.post("https://api.elevenlabs.io/v1/speech-to-text", formData, {
      headers: {
        "xi-api-key": process.env.ELEVENLABS_API_KEY,
        ...formData.getHeaders(),
      },
    });

    const transcript = response.data?.text?.trim();
    console.log("📄 STT Transcript:", transcript);
    return transcript;
  } catch (error) {
    console.error("Error in STT:", error.response?.data || error.message);
    return null;
  }
};

// 🎯 Core Function
const run = async (defaultAudioUrl, userAudioUrl) => {
  const audioFile1 = "audio1.wav";
  const audioFile2 = "audio2.wav";
  const noiseSuppressedAudio = "noise_suppressed_audio.wav";

  try {
    await downloadAudio(defaultAudioUrl, audioFile1);
    await downloadAudio(userAudioUrl, audioFile2);

    // Noise reduction
    await new Promise((resolve, reject) => {
      ffmpeg(audioFile2)
        .output(noiseSuppressedAudio)
        .audioFilters("afftdn=nf=-25")
        .on("end", resolve)
        .on("error", reject)
        .run();
    });

    // 🔍 STT speech check (before acoustic comparison)
    const transcript = await transcribeAudio(noiseSuppressedAudio);
    if (!transcript || transcript.length < 2) {
      console.log("No speech detected in user audio.");
      return {
        audioComparison: {
          mfccDistance: Infinity,
          chromaDistance: Infinity,
          zcr: Infinity,
        },
        weightedSimilarity: 100,
      };
    }

    const features1 = await extractAudioFeatures(audioFile1);
    const features2 = await extractAudioFeatures(noiseSuppressedAudio);
    deleteFileIfExists(audioFile2);
    deleteFileIfExists(noiseSuppressedAudio);

    if (detectSilentOrLowVoiceAudio(features2)) {
      return {
        audioComparison: {
          mfccDistance: Infinity,
          chromaDistance: Infinity,
          zcr: Infinity,
        },
        weightedSimilarity: 100,
      };
    }

    const audioComparison = compareAudioFeatures(features1, features2);
    const weightedSimilarity = stentWeightedAudioSimilarity(
      audioComparison.mfccDistance,
      audioComparison.chromaDistance,
      audioComparison.zcr
    );

    return {
      audioComparison,
      weightedSimilarity,
    };
  } catch (err) {
    console.error("💥 Error during comparison:", err.message);
    return {
      audioComparison: {
        mfccDistance: Infinity,
        chromaDistance: Infinity,
        zcr: Infinity,
      },
      weightedSimilarity: 100,
    };
  }
};

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
  try {
    const comparisonResults = [];
    let totalScore = 0;

    for (let i = 0; i < defaultAudios.length; i++) {
      const userAudioUrl = fileUrls[`AudioURL${i + 1}`];
      const defaultAudioUrl = defaultAudios[i];

      const result = await run(defaultAudioUrl, userAudioUrl);
      const isCorrect = result.weightedSimilarity <= similarityThreshold;
      if (isCorrect) totalScore += 1;

      comparisonResults.push({
        ItemCode: `Itemcode${i + 1}`,
        mfccDistance: result.audioComparison.mfccDistance,
        chromaDistance: result.audioComparison.chromaDistance,
        zcr: result.audioComparison.zcr,
        stentWeightedSimilarity: result.weightedSimilarity,
        Remarks: isCorrect ? "Correct" : "Incorrect",
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
  } catch (error) {
    console.error("🚨 Error saving comparison result:", error);
    throw error;
  }
};

module.exports = {
  runComparisonAndSaveResult,
};
