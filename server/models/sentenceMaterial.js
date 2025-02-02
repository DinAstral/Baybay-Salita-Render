const mongoose = require("mongoose");
const { Schema } = mongoose;

const materialSentenceSchema = new Schema(
  {
    ItemCode: {
      type: String,
      unique: true,
    },
    Type: String,
    Title: String,
    Sentence: String,
    Question1: String,
    Question2: String,
    Question3: String,
    Question4: String,
    Question5: String,
    Answer1: String,
    Answer2: String,
    Answer3: String,
    Answer4: String,
    Answer5: String,
  },
  { timestamps: true }
);

const MaterialSentenceModel = mongoose.model(
  "materials_sentence",
  materialSentenceSchema
);

module.exports = MaterialSentenceModel;
