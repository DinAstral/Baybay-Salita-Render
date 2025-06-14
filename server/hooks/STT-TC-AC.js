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
  return 0.5 * mfcc + 0.1 * chroma + 0.4 * zcr;
};

const transcribeAudio = async (filePath) => {
  try {
    const formData = new FormData();
    formData.append("file", fs.createReadStream(filePath));
    formData.append("model_id", "scribe_v1");
    formData.append("language_code", "fil");

    const response = await axios.post(
      "https://api.elevenlabs.io/v1/speech-to-text",
      formData,
      {
        headers: {
          "xi-api-key": process.env.ELEVENLABS_API_KEY,
          ...formData.getHeaders(),
        },
        validateStatus: () => true,
      }
    );

    let transcript = response.data?.text?.trim();
    if (!transcript || transcript.length < 2) {
      console.log("🚫 Transcript is empty or too short:", transcript);
      return null;
    }

    const normalizedTranscript = transcript.toLowerCase().trim();
    const cleanedTranscript = normalizedTranscript.replace(/[.!?]$/, '');

    const allowedTerms = [
      "a", "b", "d", "e", "g", "h", "i", "k", "l", "m", "n", "ng", "o", "p", "r", "s", "t", "u", "w", "y",
      "ba", "bi", "bo", "bu", "ka", "ke", "ki", "ko", "ku", "da", "de", "di", "do", "du",
      "ga", "ge", "gi", "go", "gu", "ha", "he", "hi", "ho", "hu", "la", "le", "li", "lo", "lu",
      "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu", "nga", "nge", "ngi", "ngo", "ngu",
      "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro", "ru", "sa", "se", "si", "so", "su",
      "ta", "te", "ti", "to", "tu", "wa", "we", "wi", "wo", "wu", "ya", "ye", "yi", "yo", "yu",
      "apa", "buhok", "buhuk", "bohok", "bulaklak", "bolaklak", "daliri", "kama", "kamay", "kuko", "koko", "kuku",
      "labi", "labe", "malakas", "bola", "bula", "aklat", "upuan", "lamesa", "lamisa", "ate", "baso", "bata",
      "bibe", "bebe", "bibi", "bote", "bute", "buko", "boko", "buku", "daga", "dila", "dela", "dugo", "dogo",
      "dugu", "keso", "kuya", "koya", "lobo", "lubo", "lubu", "lola", "lula", "lolo", "lulo", "lulu", "lolu",
      "mama", "mata", "misa", "mesa", "paa", "papa", "puso", "relo", "siko", "seko", "seku", "siku", "tasa",
      "toro", "toru", "turo", "turu", "ulo", "olu", "ulu", "olo", "butiki", "botiki", "buteki", "buteke",
      "daliri", "dalere", "dalire", "daleri", "gitara", "getara", "lababo", "lababu", "modelo", "mudelu",
      "modilo", "mudilu", "musika", "mosika", "museka", "moseka", "payaso", "payasu", "pisara", "pesara",
      "raketa", "rakita", "regalo", "rigalo", "salapi", "salape"
    ];

    const isValid = allowedTerms.includes(cleanedTranscript);

    if (!isValid) {
      console.log("🚫 Transcript not in allowed terms:", transcript);
      return null;
    }

    console.log("✅ Valid Transcript:", transcript);
    return transcript;
  } catch (error) {
    console.error("Error in STT:", error.response?.data || error.message);
    return null;
  }
};

const run = async (defaultAudioUrl, userAudioUrl) => {
  const audioFile1 = "audio1.wav";
  const audioFile2 = "audio2.wav";
  const noiseSuppressedAudio = "noise_suppressed_audio.wav";

  try {
    await downloadAudio(defaultAudioUrl, audioFile1);
    await downloadAudio(userAudioUrl, audioFile2);

    await new Promise((resolve, reject) => {
      ffmpeg(audioFile2)
        .output(noiseSuppressedAudio)
        .audioFilters("afftdn=nf=-25")
        .on("end", resolve)
        .on("error", reject)
        .run();
    });

    const features2 = await extractAudioFeatures(noiseSuppressedAudio);
    const isSilent = detectSilentOrLowVoiceAudio(features2);

    if (isSilent) {
      console.log("🚫 Silent/low voice detected.");
      return {
        audioComparison: {
          mfccDistance: Infinity,
          chromaDistance: Infinity,
          zcr: Infinity,
        },
        weightedSimilarity: 100,
        transcript: null,
      };
    }

    const transcript = await transcribeAudio(noiseSuppressedAudio);
    console.log("🎤 Final Transcript Output for logs:", transcript);

    if (!transcript) {
      return {
        audioComparison: {
          mfccDistance: Infinity,
          chromaDistance: Infinity,
          zcr: Infinity,
        },
        weightedSimilarity: 100,
        transcript,
      };
    }

    const features1 = await extractAudioFeatures(audioFile1);
    deleteFileIfExists(audioFile2);
    deleteFileIfExists(noiseSuppressedAudio);

    const audioComparison = compareAudioFeatures(features1, features2);
    const weightedSimilarity = stentWeightedAudioSimilarity(
      audioComparison.mfccDistance,
      audioComparison.chromaDistance,
      audioComparison.zcr
    );

    return {
      audioComparison,
      weightedSimilarity,
      transcript,
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
      transcript: null,
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
  similarityThreshold = 50
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

      const resultObject = {
        ItemCode: `Itemcode${i + 1}`,
        mfccDistance: result.audioComparison.mfccDistance,
        chromaDistance: result.audioComparison.chromaDistance,
        zcr: result.audioComparison.zcr,
        stentWeightedSimilarity: result.weightedSimilarity,
        Transcript: result.transcript,
        Remarks: isCorrect ? "Correct" : "Incorrect",
      };

      console.log(`✅ Completed comparison for Item ${i + 1}:`, resultObject);

      comparisonResults.push(resultObject);
    }

    await CompareModel.create({
      UserInputId,
      ActivityCode,
      LRN,
      Section,
      Type,
      Results: comparisonResults,
    });

    console.log("📝 Final Results Summary:", {
      score: totalScore,
      resultsWithRemarks: comparisonResults,
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
