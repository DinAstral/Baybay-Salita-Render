const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const performanceSchema = new Schema(
  {
    UserInputId: {
      type: String,
      unique: true,
    },
    ActivityCode: String,
    LRN: String,
    Section: String,
    Type: String,
    TimeRead: String,
    Score: Object,
    Result: String,
    Story: String,
    Title: String,
    QuestionData: Object,
    PerformanceItems: Object,
  },
  { timestamps: true }
);

const PerformanceModel = mongoose.model(
  "student_performance",
  performanceSchema
);

module.exports = PerformanceModel;
