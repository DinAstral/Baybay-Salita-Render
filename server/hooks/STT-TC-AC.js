const axios = require("axios");
const Meyda = require("meyda");
const fs = require("fs");
const fsp = require("fs").promises;
const ffmpeg = require("fluent-ffmpeg");
const wav = require("wav-decoder");
const mongoose = require("mongoose");
const CompareModel = require("../models/ComparisonResult");
const AssemblyAI = require("assemblyai");
const assembly = new AssemblyAI({ apiKey: process.env.ASSEMBLYAI_API_KEY });

// Function to download and save audio files locally
const downloadAudio = async (url, filename) => {
  const response = await axios({ method: "GET", url, responseType: "stream" });
  return new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(filename);
    response.data.pipe(writer);
    writer.on("finish", resolve);
    writer.on("error", reject);
  });
};

// Function to delete a file if it exists
const deleteFileIfExists = (filePath) => {
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
};

// Function to chunk audio
const chunkArray = (arr, chunkSize) => {
  const chunks = [];
  for (let i = 0; i < arr.length; i += chunkSize) {
    const chunk = arr.slice(i, i + chunkSize);
    if (chunk.length < chunkSize) {
      const padded = new Float32Array(chunkSize);
      padded.set(chunk);
      chunks.push(padded);
    } else {
      chunks.push(chunk);
    }
  }
  return chunks;
};

// Validate with AssemblyAI if audio contains speech in Filipino
const validateSpeechContent = async (audioFilePath) => {
  try {
    const audioData = await fsp.readFile(audioFilePath);
    const transcript = await assembly.transcripts.transcribe({
      audio: audioData,
      language_code: "fil",
    });

    if (transcript.text && transcript.text.trim().length > 0) {
      console.log("May natukoy na speech sa audio.");
      return true;
    } else {
      console.log("Walang natukoy na speech sa audio.");
      return false;
    }
  } catch (error) {
    console.error("Error sa STT validation:", error.message);
    return false;
  }
};

// Detect low voice
const detectSilentOrLowVoiceAudio = (features) => {
  const zcrThreshold = 10.0;
  const energyThreshold = 0.03;
  const avgZCR = features.zcr.reduce((a, b) => a + b, 0) / features.zcr.length;
  const avgEnergy = features.energy ?? 0;

  const isSilent = avgZCR < zcrThreshold && avgEnergy < energyThreshold;
  console.log("Is Silent or Low Voice Detected:", isSilent);
  return isSilent;
};

// Extract audio features
const extractAudioFeatures = async (audioPath) => {
  return new Promise((resolve, reject) => {
    const tempOutput = `temp_${audioPath}`;
    deleteFileIfExists(tempOutput);

    ffmpeg(audioPath)
      .toFormat("wav")
      .on("end", async () => {
        try {
          const data = await fsp.readFile(tempOutput);
          const audioData = await wav.decode(data);
          const channelData = audioData.channelData[0];
          const bufferSize = 512;
          const audioChunks = chunkArray(channelData, bufferSize);

          const totalChunks = audioChunks.length;
          const sumFeatures = audioChunks.reduce((acc, chunk, idx) => {
            const features = Meyda.extract(["mfcc", "chroma", "zcr"], chunk);
            if (idx === 0) {
              for (let key in features) {
                acc[key] = Array.isArray(features[key])
                  ? features[key].slice()
                  : [features[key]];
              }
            } else {
              for (let key in features) {
                if (Array.isArray(features[key])) {
                  acc[key] = acc[key].map((v, i) => v + features[key][i]);
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
      .on("error", (err) => reject(err))
      .save(tempOutput);
  });
};

// Compare features
const compareAudioFeatures = (f1, f2) => {
  const euclideanDistance = (v1, v2) =>
    Math.sqrt(v1.reduce((sum, val, i) => sum + Math.pow(val - v2[i], 2), 0));

  return {
    mfccDistance: euclideanDistance(f1.mfcc, f2.mfcc),
    chromaDistance: euclideanDistance(f1.chroma, f2.chroma),
    zcr: euclideanDistance(f1.zcr, f2.zcr),
  };
};

// Weighted scoring
const stentWeightedAudioSimilarity = (mfcc, chroma, zcr) => {
  const weightMfcc = 0.4;
  const weightChroma = 0.3;
  const weightZcr = 0.3;
  return weightMfcc * mfcc + weightChroma * chroma + weightZcr * zcr;
};

// Main runner
const run = async (defaultAudioUrl, userAudioUrl) => {
  try {
    const audioFile1 = "audio1.wav";
    const audioFile2 = "audio2.wav";
    const noiseSuppressedAudio = "noise_suppressed_audio.wav";

    await downloadAudio(defaultAudioUrl, audioFile1);
    await downloadAudio(userAudioUrl, audioFile2);

    const hasSpeech = await validateSpeechContent(audioFile2);
    if (!hasSpeech) {
      console.log("Walang speech detected. Itinuturing na mali.");
      return {
        audioComparison: null,
        weightedSimilarity: 100,
      };
    }

    // Suppress noise
    await new Promise((resolve, reject) => {
      ffmpeg(audioFile2)
        .output(noiseSuppressedAudio)
        .audioFilters("afftdn=nf=-25")
        .on("end", resolve)
        .on("error", reject)
        .run();
    });

    const features1 = await extractAudioFeatures(audioFile1);
    const features2 = await extractAudioFeatures(noiseSuppressedAudio);

    deleteFileIfExists(audioFile2);
    deleteFileIfExists(noiseSuppressedAudio);

    if (detectSilentOrLowVoiceAudio(features2)) {
      console.log("Low voice detected. Itinuturing na mali.");
      return {
        audioComparison: null,
        weightedSimilarity: 100,
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
    };
  } catch (err) {
    console.error("Error sa paghahambing:", err.message);
    throw err;
  }
};

// Comparison & saving
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
    const results = [];
    let totalScore = 0;

    for (let i = 0; i < defaultAudios.length; i++) {
      const userAudioUrl = fileUrls[`AudioURL${i + 1}`];
      const defaultAudioUrl = defaultAudios[i];
      const result = await run(defaultAudioUrl, userAudioUrl);

      const isCorrect = result.weightedSimilarity <= similarityThreshold;
      if (isCorrect) totalScore++;

      results.push({
        ItemCode: `Itemcode${i + 1}`,
        mfccDistance: result.audioComparison?.mfccDistance ?? null,
        chromaDistance: result.audioComparison?.chromaDistance ?? null,
        zcr: result.audioComparison?.zcr ?? null,
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
      Results: results,
    });

    return { score: totalScore, resultsWithRemarks: results };
  } catch (error) {
    console.error("Error sa final comparison:", error);
    throw error;
  }
};

module.exports = {
  runComparisonAndSaveResult,
};
