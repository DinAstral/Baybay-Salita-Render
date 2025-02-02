const Student = require("../models/student");
const mongoose = require("mongoose");
const jwt = require("jsonwebtoken");

const mobileLogin = async (req, res) => {
  try {
    // Extract firstName and lastName from req.body
    const { FirstName, LastName } = req.body;

    // Check if both firstName and lastName are provided
    if (!FirstName || !LastName) {
      return res.status(400).json({
        error: "First Name and Last Name are required.",
      });
    }

    // Combine firstName and lastName for display or further use
    const fullName = `${FirstName} ${LastName}`;

    // Find user by firstName and lastName (modify query based on your DB schema)
    const user = await Student.findOne({
      FirstName: FirstName,
      LastName: LastName,
    });

    if (!user) {
      return res.status(404).json({
        error: "User not found.",
      });
    }

    // Generate JWT token
    jwt.sign(
      {
        LRN: user.LRN,
        FirstName: user.FirstName,
        LastName: user.LastName,
      },
      process.env.JWT_SECRET,
      {},
      (err, token) => {
        if (err) {
          console.error(err);
          return res.status(500).json({ error: "Failed to generate token" });
        }
        // Set cookie with token
        res
          .cookie("token", token, { httpOnly: true })
          .json({ success: true, token, role: user.role, user });
      }
    );
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

// Gets the data of the Student base on the _id
const getStudentbyLRN = async (req, res) => {
  const { LRN } = req.params;

  try {
    const student = await Student.findOne({ LRN });
    if (!student) {
      return res.status(404).json({
        message: "No account found",
      });
    }
    res.status(200).json(student);
  } catch (error) {
    res.status(500).json({
      message: error.message,
    });
  }
};

module.exports = {
  mobileLogin,
  getStudentbyLRN,
};
