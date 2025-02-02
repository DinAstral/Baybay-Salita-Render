const mongoose = require("mongoose");
const Material = require("../models/materials");
const SentenceModel = require("../models/sentenceMaterial");
const Performance = require("../models/performance");
const AssessmentModel = require("../models/assessment");
const cloudinary = require("cloudinary").v2;
const path = require("path");
const multer = require("multer");

const { wordUpload, UserUpload } = require("../middleware/multer");
const {
  cloudinaryUploader,
  cloudinaryUploaderUser,
} = require("../hooks/cloudinary");

cloudinary.config({
  cloud_name: "dvcqnbkwb",
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true,
});

// Utility functions to generate random codes
function generateRandomCodeItem(length) {
  const characters = "0123456789";
  let result = "item_";
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

function generateRandomCodeSentence(length) {
  const characters = "0123456789";
  let result = "sentence_";
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

function generateRandomCodeUser(length) {
  const characters = "0123456789";
  let result = "UserInput_";
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

function generateRandomCode(length) {
  const characters = "0123456789";
  let result = "assessment_";
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

// Submit assessment
const submitAssessment = async (req, res) => {
  const randomCode = generateRandomCode(6);
  try {
    const {
      UserID,
      Period,
      Section,
      Type,
      Title,
      Item1,
      Item2,
      Item3,
      Item4,
      Item5,
      Item6,
      Item7,
      Item8,
      Item9,
      Item10,
    } = req.body;

    if (!Period || !Type) {
      return res.json({ error: "All fields are required" });
    }

    const exist = await AssessmentModel.findOne({ ActivityCode: randomCode });
    if (exist) {
      return res.json({ error: "ActivityCode is already taken" });
    }

    // Fetch materials for each item
    const items = [
      Item1,
      Item2,
      Item3,
      Item4,
      Item5,
      Item6,
      Item7,
      Item8,
      Item9,
      Item10,
    ];
    const materials = await Material.find({ ItemCode: { $in: items } });

    const assessmentItems = items.map((itemCode) => {
      const material = materials.find((m) => m.ItemCode === itemCode);
      return {
        ItemCode: itemCode,
        Word: material?.Word || "",
        Image: material?.Image || "",
        SecureImage: material?.SecureImage || "",
        Audio: material?.Audio || "",
        SecureAudio: material?.SecureAudio || "",
      };
    });

    // Fetch the corresponding sentence for Pagbabasa type assessment
    let questions = [];
    let sentenceText = ""; // Initialize the sentence to save later

    if (Type === "Pagbabasa") {
      const sentence = await SentenceModel.findOne({ Title }); // Fetch the sentence based on the Title

      if (sentence) {
        sentenceText = sentence.Sentence; // Store the Sentence for saving

        questions = [
          {
            ItemCode: sentence.ItemCode,
            Question: sentence.Question1,
            Answer: sentence.Answer1,
          },
          {
            ItemCode: sentence.ItemCode,
            Question: sentence.Question2,
            Answer: sentence.Answer2,
          },
          {
            ItemCode: sentence.ItemCode,
            Question: sentence.Question3,
            Answer: sentence.Answer3,
          },
          {
            ItemCode: sentence.ItemCode,
            Question: sentence.Question4,
            Answer: sentence.Answer4,
          },
          {
            ItemCode: sentence.ItemCode,
            Question: sentence.Question5,
            Answer: sentence.Answer5,
          },
        ];
      }
    }

    const act = await AssessmentModel.create({
      UserID,
      ActivityCode: randomCode,
      Section,
      Period,
      Type,
      Title,
      Sentence: sentenceText, // Save the sentence if Pagbabasa is selected
      Questions: questions, // Saving grouped questions and answers
      Items: assessmentItems, // Save items with image and audio
      Assessment: "Submitted",
    });

    return res.json(act);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" });
  }
};

// get assessment based on Activitycode
const getAssessmentCode = async (req, res) => {
  const { ActivityCode } = req.params;

  try {
    const activity = await AssessmentModel.findOne({ ActivityCode });

    // If no parent is found, return a 404 status
    if (!activity) {
      return res.json({
        message: "No activity found",
      });
    }

    // Return the found parent
    res.json(activity);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

// get assessment based on Activitycode
const getAssessmentID = async (req, res) => {
  const { UserID } = req.params;

  try {
    const activity = await AssessmentModel.find({ UserID });

    // If no assessment is found, return a 404 status
    if (!activity) {
      return res.json({
        message: "No activity found",
      });
    }

    // Return the found assessment
    res.json(activity);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

// get performance based on input ID
const getOnePerformance = async (req, res) => {
  const { UserInputId } = req.params;

  try {
    const activity = await Performance.findOne({ UserInputId });

    // If no parent is found, return a 404 status
    if (!activity) {
      return res.json({
        message: "No user input found",
      });
    }

    // Return the found parent
    res.json(activity);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

// get performance based on input LRN getPerformanceStudent
const getPerformanceStudent = async (req, res) => {
  const { LRN } = req.params;

  try {
    const activity = await Performance.find({ LRN });

    // If no parent is found, return a 404 status
    if (!activity) {
      return res.json({
        message: "No LRN input found",
      });
    }

    // Return the found parent
    res.json(activity);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

// Get performance data
const getPerformance = (req, res) => {
  Performance.find()
    .then((users) => res.json(users))
    .catch((err) => res.json(err));
};

// Get Sentence data
const getSentence = (req, res) => {
  SentenceModel.find()
    .then((users) => res.json(users))
    .catch((err) => res.json(err));
};

// Delete assessment
const deleteAssessment = async (req, res) => {
  const { id } = req.params;

  if (!mongoose.Types.ObjectId.isValid(id)) {
    return res.status(400).json({ message: "Invalid activity ID format" });
  }

  try {
    const activity = await AssessmentModel.findByIdAndDelete(id);

    if (!activity) {
      return res.status(404).json({ message: "No activity found" });
    }

    return res
      .status(200)
      .json({ message: "Activity deleted successfully", data: activity });
  } catch (error) {
    return res
      .status(500)
      .json({ message: "Error deleting activity", error: error.message });
  }
};

// Delete performance
const deletePerformance = async (req, res) => {
  const { id } = req.params;

  if (!mongoose.Types.ObjectId.isValid(id)) {
    return res.status(400).json({ message: "Invalid performance ID format" });
  }

  try {
    const activity = await Performance.findByIdAndDelete(id);

    if (!activity) {
      return res.status(404).json({ message: "No student performance found" });
    }

    return res.status(200).json({
      message: "Student performance deleted successfully",
      data: activity,
    });
  } catch (error) {
    return res.status(500).json({
      message: "Error deleting Student performance",
      error: error.message,
    });
  }
};

const importWord = async (req, res) => {
  const itemID = generateRandomCodeItem(6); // Assume this function is defined somewhere

  wordUpload(req, res, async function (err) {
    if (err instanceof multer.MulterError) {
      return res.json({ error: `Multer Error: ${err.message}` });
    } else if (err) {
      return res.json({ error: `Error: ${err.message}` });
    }

    try {
      // Extracting Type and Word from request body
      const { Type, Word } = req.body;
      if (!Type || !Word) {
        return res.json({ error: "Type and Word are required" });
      }

      // Uploading to Cloudinary
      const uploadResponse = await cloudinaryUploader(req, res);
      const imageFile = uploadResponse.uploadImage.url;
      const securedImageFile = uploadResponse.uploadImage.secure_url;
      const audioFile = uploadResponse.uploadAudio.url;
      const securedAudioFile = uploadResponse.uploadAudio.secure_url;

      // Creating a new Material object
      const material = new Material({
        ItemCode: itemID,
        Type,
        Word,
        Image: imageFile,
        SecureImage: securedImageFile,
        Audio: audioFile,
        SecureAudio: securedAudioFile,
      });

      // Saving material to the database
      await material.save();

      return res.json({ message: "Word and files successfully uploaded" });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Server error" });
    }
  });
};

const importSentence = async (req, res) => {
  const itemID = generateRandomCodeSentence(6); // Generate unique item code

  try {
    // Extracting Type, Sentence, and Questions from request body
    const {
      Type,
      Title,
      Sentence,
      Question1,
      Question2,
      Question3,
      Question4,
      Question5,
      Answer1,
      Answer2,
      Answer3,
      Answer4,
      Answer5,
    } = req.body;

    // Basic field validation
    if (!Type || typeof Type !== "string" || Type.trim() === "") {
      return res.json({
        error: "Type is required and must be a non-empty string",
      });
    }

    if (!Title || typeof Title !== "string" || Title.trim() === "") {
      return res.json({
        error: "Title is required and must be a non-empty string",
      });
    }

    if (!Sentence || typeof Sentence !== "string" || Sentence.trim() === "") {
      return res.json({
        error: "Sentence is required and must be a non-empty string",
      });
    }

    // Ensure all questions are present and non-empty
    if (
      !Question1 ||
      typeof Question1 !== "string" ||
      Question1.trim() === ""
    ) {
      return res.json({
        error: "Question1 is required and must be a non-empty string",
      });
    }
    if (
      !Question2 ||
      typeof Question2 !== "string" ||
      Question2.trim() === ""
    ) {
      return res.json({
        error: "Question2 is required and must be a non-empty string",
      });
    }
    if (
      !Question3 ||
      typeof Question3 !== "string" ||
      Question3.trim() === ""
    ) {
      return res.json({
        error: "Question3 is required and must be a non-empty string",
      });
    }
    if (
      !Question4 ||
      typeof Question4 !== "string" ||
      Question4.trim() === ""
    ) {
      return res.json({
        error: "Question4 is required and must be a non-empty string",
      });
    }
    if (
      !Question5 ||
      typeof Question5 !== "string" ||
      Question5.trim() === ""
    ) {
      return res.json({
        error: "Question5 is required and must be a non-empty string",
      });
    }

    // Ensure all answers are present and non-empty
    if (!Answer1 || typeof Answer1 !== "string" || Answer1.trim() === "") {
      return res.json({
        error:
          "Answer1 is required for Question1 and must be a non-empty string",
      });
    }
    if (!Answer2 || typeof Answer2 !== "string" || Answer2.trim() === "") {
      return res.json({
        error:
          "Answer2 is required for Question2 and must be a non-empty string",
      });
    }
    if (!Answer3 || typeof Answer3 !== "string" || Answer3.trim() === "") {
      return res.json({
        error:
          "Answer3 is required for Question3 and must be a non-empty string",
      });
    }
    if (!Answer4 || typeof Answer4 !== "string" || Answer4.trim() === "") {
      return res.json({
        error:
          "Answer4 is required for Question4 and must be a non-empty string",
      });
    }
    if (!Answer5 || typeof Answer5 !== "string" || Answer5.trim() === "") {
      return res.json({
        error:
          "Answer5 is required for Question5 and must be a non-empty string",
      });
    }

    // Create a new SentenceModel object
    const material = new SentenceModel({
      ItemCode: itemID,
      Type,
      Title,
      Sentence,
      Question1,
      Question2,
      Question3,
      Question4,
      Question5,
      Answer1,
      Answer2,
      Answer3,
      Answer4,
      Answer5,
    });

    // Save the material to the database
    await material.save();

    return res.json({ message: "Sentence successfully uploaded" });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const { runComparisonAndSaveResult } = require("../hooks/STT-TC-AC"); // Import the helper function for comparison

const userInputAudio = async (req, res) => {
  const InputID = generateRandomCodeUser(6);

  UserUpload(req, res, async function (err) {
    if (err instanceof multer.MulterError) {
      return res.json({ error: `${err.message}` });
    } else if (err) {
      return res.json({ error: `${err.message}` });
    }

    try {
      const {
        Type,
        LRN,
        Section,
        ActivityCode,
        Itemcode1,
        Itemcode2,
        Itemcode3,
        Itemcode4,
        Itemcode5,
        Itemcode6,
        Itemcode7,
        Itemcode8,
        Itemcode9,
        Itemcode10,
      } = req.body;

      // Upload user input audio files to cloudinary
      const uploadResponse = await cloudinaryUploaderUser(req, res);

      // Get uploaded URLs for each user input audio
      const fileUrls = {
        AudioURL1: uploadResponse.uploadAudioUser1
          ? uploadResponse.uploadAudioUser1.secure_url
          : "",
        AudioURL2: uploadResponse.uploadAudioUser2
          ? uploadResponse.uploadAudioUser2.secure_url
          : "",
        AudioURL3: uploadResponse.uploadAudioUser3
          ? uploadResponse.uploadAudioUser3.secure_url
          : "",
        AudioURL4: uploadResponse.uploadAudioUser4
          ? uploadResponse.uploadAudioUser4.secure_url
          : "",
        AudioURL5: uploadResponse.uploadAudioUser5
          ? uploadResponse.uploadAudioUser5.secure_url
          : "",
        AudioURL6: uploadResponse.uploadAudioUser6
          ? uploadResponse.uploadAudioUser6.secure_url
          : "",
        AudioURL7: uploadResponse.uploadAudioUser7
          ? uploadResponse.uploadAudioUser7.secure_url
          : "",
        AudioURL8: uploadResponse.uploadAudioUser8
          ? uploadResponse.uploadAudioUser8.secure_url
          : "",
        AudioURL9: uploadResponse.uploadAudioUser9
          ? uploadResponse.uploadAudioUser9.secure_url
          : "",
        AudioURL10: uploadResponse.uploadAudioUser10
          ? uploadResponse.uploadAudioUser10.secure_url
          : "",
      };

      // Group item codes for fetching from the database
      const itemCodes = [
        Itemcode1,
        Itemcode2,
        Itemcode3,
        Itemcode4,
        Itemcode5,
        Itemcode6,
        Itemcode7,
        Itemcode8,
        Itemcode9,
        Itemcode10,
      ];

      // Fetch words and default audio for each ItemCode from the database
      const materials = await Material.find({ ItemCode: { $in: itemCodes } });

      // Create an array of items with their corresponding data
      const assessmentItems = itemCodes.map((itemCode, index) => {
        const material = materials.find((m) => m.ItemCode === itemCode);
        return {
          ItemCode: itemCode,
          Word: material?.Word || "",
          Audio: material?.Audio || "",
          SecureAudio: material?.SecureAudio || "",
          UserAudioURL: fileUrls[`AudioURL${index + 1}`],
        };
      });

      // Run audio comparison and save the results to the database, get the score and remarks
      const { score, resultsWithRemarks } = await runComparisonAndSaveResult(
        InputID,
        ActivityCode,
        LRN,
        Section,
        Type,
        fileUrls,
        assessmentItems.map((item) => item.Audio) // Extract default audios
      );

      // Prepare the data for insertion into the Performance model
      const performanceData = {
        UserInputId: InputID,
        ActivityCode,
        LRN,
        Section,
        Type,
        Score: score, // Store the total score
        Result: "Submitted",
        PerformanceItems: resultsWithRemarks.map((result, index) => ({
          ItemCode: result.ItemCode,
          Word: assessmentItems[index].Word,
          UserAudioURL: assessmentItems[index].UserAudioURL,
          DefaultAudio: assessmentItems[index].Audio,
          SecureAudio: assessmentItems[index].SecureAudio,
          Remarks: result.Remarks, // Add Remarks field to PerformanceItems
        })),
      };

      // Insert data into the Performance model
      const insert = await Performance.create(performanceData);

      if (insert) {
        await AssessmentModel.updateOne(
          { ActivityCode },
          { $addToSet: { completedBy: LRN } }
        );

        return res.json({
          message:
            "Audio files uploaded, data stored, and comparison results saved successfully.",
          UserInputId: InputID, // Include UserInputId in response
        });
      } else {
        return res.status(500).json({
          error: "Upload unsuccessful. Please try again later!",
        });
      }
    } catch (error) {
      console.error("Error during upload process:", error);
      if (!res.headersSent) {
        return res
          .status(500)
          .json({ error: "An error occurred during the upload process." });
      }
    }
  });
};

const userInputSentence = async (req, res) => {
  const InputID = generateRandomCodeUser(6);

  try {
    const {
      Type,
      LRN,
      Section,
      ActivityCode,
      TimeRead,
      Score,
      Story,
      Title,
      QuestionData,
    } = req.body;

    // Prepare the data for insertion into the Performance model
    const performanceData = {
      UserInputId: InputID,
      ActivityCode,
      LRN,
      Section,
      Type,
      TimeRead,
      Story,
      Title,
      QuestionData,
      Score,
      Result: "Submitted",
    };

    // Insert data into the Performance model
    const insert = await Performance.create(performanceData);

    if (insert) {
      await AssessmentModel.updateOne(
        { ActivityCode },
        { $addToSet: { completedBy: LRN } }
      );

      return res.json({
        message: "Data stored save successfully.",
      });
    } else {
      return res.status(500).json({
        error: "Upload unsuccessful. Please try again later!",
      });
    }
  } catch (error) {
    console.error("Error during upload process:", error);
    if (!res.headersSent) {
      return res
        .status(500)
        .json({ error: "An error occurred during the upload process." });
    }
  }
};

const getAssessmentIncomplete = async (req, res) => {
  const { lrn, section } = req.query;

  try {
    const assessments = await AssessmentModel.find({
      Section: section,
      completedBy: { $ne: lrn },
    }).lean();

    return res.status(200).json(assessments);
  } catch (error) {
    console.error("Error fetching incomplete assessments:", error);
    res.status(500).json({ error: "Failed to fetch assessments." });
  }
};

const getImportWords = (req, res) => {
  Material.find()
    .then((users) => res.json(users))
    .catch((err) => res.json(err));
};

module.exports = {
  importWord,
  importSentence,
  submitAssessment,
  getAssessmentCode,
  getAssessmentID,
  getImportWords,
  getOnePerformance,
  getPerformance,
  getSentence,
  deleteAssessment,
  deletePerformance,
  userInputAudio,
  userInputSentence,
  getPerformanceStudent,
  getAssessmentIncomplete,
};
