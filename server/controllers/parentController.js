const Parent = require("../models/parents");
const Student = require("../models/student");
const mongoose = require("mongoose");
const Feedback = require("../models/feedback");

function generateRandomCodeFeed(length) {
  const characters = "0123456789";
  let result = "feedbackID_";
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

const getParentUsers = (req, res) => {
  Parent.find()
    .then((users) => res.json(users))
    .catch((err) => res.json(err));
};

const getParentUser = async (req, res) => {
  const { UserID } = req.params; // Extract UserID from route parameters

  try {
    // Find the parent by UserID
    const user = await Parent.findOne({ UserID });

    // If no parent is found, return a 404 status
    if (!user) {
      return res.json({
        message: "No account found",
      });
    }

    // Return the found parent
    res.json(user);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

const getStudentUser = async (req, res) => {
  const { LRN } = req.params; // Extract UserID from route parameters

  try {
    // Find the student by LRN
    const user = await Student.findOne({ LRN });

    // If no student is found, return a 404 status
    if (!user) {
      return res.json({
        message: "No account found",
      });
    }

    // Return the found student
    res.json(user);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

// Submit feedback
const submitFeedback = async (req, res) => {
  const randomCode = generateRandomCodeFeed(6);
  try {
    const { UserID, LRN, Title, ActivityCode, Type, Feedback_Date, Context } =
      req.body;

    if (!Title || !Type) {
      return res.json({ error: "All fields are required" });
    }

    const feed = await Feedback.create({
      FeedbackID: randomCode,
      UserID,
      LRN,
      Title,
      ActivityCode,
      Type,
      Feedback_Date,
      Context,
    });

    return res.json(feed);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const getFeedbacks = (req, res) => {
  Feedback.find()
    .then((users) => res.json(users))
    .catch((err) => res.json(err));
};

module.exports = {
  getParentUsers,
  getParentUser,
  getStudentUser,
  submitFeedback,
  getFeedbacks,
};
