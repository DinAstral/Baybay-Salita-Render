const multer = require("multer");
const path = require("path");
const mongoose = require("mongoose");

const storage = multer.diskStorage({
  filename: (req, file, cb) => {
    let fileExt = file.originalname;
    let fileName = "";

    //Generate new filename
    if (file.fieldname === "Profile") {
      fileName = `image_${fileExt}`;
    } else if (file.fieldname === "Image") {
      fileName = `image_${fileExt}`;
    } else if (file.fieldname === "Audio") {
      fileName = `audio_${fileExt}`;
    } else {
      fileName = `user_${fileExt}`;
    }

    cb(null, fileName);
  },
});

const fileFilter = (req, file, cb) => {
  try {
    // Check for image or audio type
    if (file.fieldname === "Image" || file.fieldname === "Profile") {
      if (
        file.mimetype === "image/jpeg" ||
        file.mimetype === "image/png" ||
        file.mimetype === "image/svg+xml" // Corrected MIME type for SVG
      ) {
        return cb(null, true);
      } else {
        return cb(
          new Error("Invalid file type. Only JPEG, PNG, and SVG are allowed.")
        );
      }
    } else if (
      file.fieldname.startsWith("Audio") ||
      file.fieldname.startsWith("User")
    ) {
      if (file.mimetype.startsWith("audio/")) {
        return cb(null, true);
      } else {
        return cb(
          new Error("Invalid file type. Only audio files are allowed.")
        );
      }
    } else {
      return cb(
        new Error(
          "Invalid fieldname. Please check the fieldname used in the upload."
        )
      );
    }
  } catch (error) {
    cb(new Error("Error occurred during file upload validation."));
  }
};

// Set general file size limit to handle errors better
const limits = {
  fileSize: 100 * 1024 * 1024, // 100 MB for all files
};

// Configure the multer upload
const profileUpload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: limits.fileSize,
  },
}).fields([{ name: "Profile", maxCount: 1 }]);

const wordUpload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: limits.fileSize,
  },
}).fields([
  { name: "Image", maxCount: 1 },
  { name: "Audio", maxCount: 1 },
]);

const UserUpload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: limits.fileSize,
  },
}).fields([
  { name: "User1", maxCount: 1 },
  { name: "User2", maxCount: 1 },
  { name: "User3", maxCount: 1 },
  { name: "User4", maxCount: 1 },
  { name: "User5", maxCount: 1 },
  { name: "User6", maxCount: 1 },
  { name: "User7", maxCount: 1 },
  { name: "User8", maxCount: 1 },
  { name: "User9", maxCount: 1 },
  { name: "User10", maxCount: 1 },
]);

// Export the upload handlers
module.exports = { wordUpload, UserUpload, profileUpload };
