const cloudinary = require("cloudinary").v2;

const cloudinaryUploaderProfile = async (req, res) => {
  const { files } = req;
  const profileFile = files["Profile"] ? files["Profile"][0] : null;

  if (!profileFile) {
    return res.json({ error: "File not found" });
  }

  try {
    const uploadProfile = await cloudinary.uploader.upload(profileFile.path, {
      resource_type: "image",
      public_id: `image/${profileFile.filename}`,
      folder: "picture",
    });

    return { uploadProfile };
  } catch (error) {
    console.log(error);
    throw new Error("Cloudinary upload failed");
  }
};

const cloudinaryUploader = async (req, res) => {
  const { files } = req;
  const imageFile = files["Image"] ? files["Image"][0] : null;
  const audioFile = files["Audio"] ? files["Audio"][0] : null;

  if (!imageFile || !audioFile) {
    return res.json({ error: "File not found" });
  }

  try {
    const uploadImage = await cloudinary.uploader.upload(imageFile.path, {
      resource_type: "image",
      public_id: `image/${imageFile.filename}`,
      folder: "images",
    });

    const uploadAudio = await cloudinary.uploader.upload(audioFile.path, {
      resource_type: "auto",
      public_id: `audio/${audioFile.filename}`,
      folder: "default_audio",
    });

    return { uploadImage, uploadAudio };
  } catch (error) {
    console.log(error);
    throw new Error("Cloudinary upload failed");
  }
};

const cloudinaryUploaderUser = async (req, res) => {
  const { files } = req;
  const audioFiles = {
    User1: files["User1"] ? files["User1"][0] : null,
    User2: files["User2"] ? files["User2"][0] : null,
    User3: files["User3"] ? files["User3"][0] : null,
    User4: files["User4"] ? files["User4"][0] : null,
    User5: files["User5"] ? files["User5"][0] : null,
    User6: files["User6"] ? files["User6"][0] : null,
    User7: files["User7"] ? files["User7"][0] : null,
    User8: files["User8"] ? files["User8"][0] : null,
    User9: files["User9"] ? files["User9"][0] : null,
    User10: files["User10"] ? files["User10"][0] : null,
  };

  try {
    const uploadPromises = Object.keys(audioFiles).map(async (key) => {
      const file = audioFiles[key];
      if (file) {
        // Check if the file is empty (size == 0 bytes)
        if (file.size === 0) {
          console.log(`${key} file is empty.`);
          return `${key} file is empty`; // Skip uploading empty files
        }
        try {
          // Attempt to upload to Cloudinary if the file is not empty
          return await cloudinary.uploader.upload(file.path, {
            resource_type: "auto",
            public_id: `audioUser/${file.filename}`,
            folder: "user_audio",
          });
        } catch (uploadError) {
          console.log(`Error uploading ${key}:`, uploadError);
          return null; // Return null if there's an issue with this specific upload
        }
      }
      return `${key} file not provided`; // If no file exists, return this message
    });

    const [
      uploadAudioUser1,
      uploadAudioUser2,
      uploadAudioUser3,
      uploadAudioUser4,
      uploadAudioUser5,
      uploadAudioUser6,
      uploadAudioUser7,
      uploadAudioUser8,
      uploadAudioUser9,
      uploadAudioUser10,
    ] = await Promise.all(uploadPromises);

    return {
      uploadAudioUser1,
      uploadAudioUser2,
      uploadAudioUser3,
      uploadAudioUser4,
      uploadAudioUser5,
      uploadAudioUser6,
      uploadAudioUser7,
      uploadAudioUser8,
      uploadAudioUser9,
      uploadAudioUser10,
    };
  } catch (error) {
    console.log("Error during upload process:", error);
    throw new Error("Cloudinary upload failed");
  }
};

/*const url = cloudinary.url("", {
  transformation: [
    {
      fetch_format: "auto",
    },
    {
      quality: "auto",
    },
  ],
});

(async function () {
  try {
    const results = await cloudinary.uploader.upload("");
    console.log(results);
    const url = cloudinary.url(results.public_id, {
      transformation: [
        {
          fetch_format: "auto",
          quality: "auto",
        },
        {
          width: "1200",
          height: "1200",
          crop: "fill",
          gravity: "auto",
        },
      ],
    });
    console.log(url);
  } catch (error) {
    console.error("Error uploading image:", error);
  }
})();*/

module.exports = {
  cloudinaryUploader,
  cloudinaryUploaderUser,
  cloudinaryUploaderProfile,
};
