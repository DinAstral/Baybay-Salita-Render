const mongoose = require("mongoose");
const { Schema } = mongoose;

const materialSchema = new Schema(
  {
    ItemCode: {
      type: String,
      unique: true,
    },
    Type: String,
    Word: String,
    Image: String,
    SecureImage: String,
    Audio: String,
    SecureAudio: String,
  },
  { timestamps: true }
);

const MaterialModel = mongoose.model("materials", materialSchema);

module.exports = MaterialModel;
