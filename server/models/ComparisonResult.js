const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const comparisonSchema = new Schema(
  {
    UserInputId: String,
    LRN: String,
    Section: String,
    ActivityCode: String,
    Type: String,
    Results: Object,
  },
  { timestamps: true }
);

const ComparisonModel = mongoose.model(
  "result_comparison_audio",
  comparisonSchema
);

module.exports = ComparisonModel;
