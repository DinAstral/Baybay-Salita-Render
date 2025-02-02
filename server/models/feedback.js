const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const FeedbackSchema = new Schema(
  {
    UserID: {
      type: String,
    },
    FeedbackID: {
      type: String,
      unique: true,
    },
    email: {
      type: String,
    },
    LRN: String,
    Title: String,
    ActivityCode: String,
    Type: String,
    Feedback_Date: String,
    Context: String,
  },
  { timestamps: true }
);

const FeedbackModel = mongoose.model("Feedback", FeedbackSchema);

module.exports = FeedbackModel;
