const User = require("../models/users");
const StudentModel = require("../models/student");
const Teacher = require("../models/teachers");
const Admin = require("../models/admin");
const UserOTPVerification = require("../models/UserOTPVerification");
const { hashPassword, comparePassword } = require("../helpers/auth");
const jwt = require("jsonwebtoken");
const mongoose = require("mongoose");
const nodemailer = require("nodemailer");
const bcrypt = require("bcryptjs");
const ParentModel = require("../models/parents");

let transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  auth: {
    user: process.env.AUTH_EMAIL,
    pass: process.env.AUTH_PASS,
  },
});

// Send verification email
const sendVerificationEmail = async ({ UserID, email }, res) => {
  try {
    const otp = `${Math.floor(100000 + Math.random() * 900000)}`;

    const mailOptions = {
      from: process.env.AUTH_EMAIL,
      to: email,
      subject: "Verify Your Email",
      html: `<p><i>Mabuhay!</i></p>
              <p>We received a registration request for your account. To verify your account, please use the following One-Time Password (OTP) code.</p>
              <h3>OTP: ${otp}</h3>
              <p>This code will <b>expire in 1 hour.</b></p>
              <h4>Link: <a href="https://baybay-salita-heroku-8c328f3ddd0f.herokuapp.com/verifyEmail">Verify your email</a> </h4>
              <p>If you didn't initiate this request, please disregard this email.</p>
              <p>Thank you.</p>
              <br>
              <p>Best regards,<br>
              <b>BaybaySalita</b></p>`,
    };

    // Hash OTP
    const hashedOTP = await bcrypt.hash(otp, 10);

    // Save OTP to DB
    const newOTPVerification = new UserOTPVerification({
      userId: UserID,
      otp: hashedOTP,
      createdAt: Date.now(),
      expiresAt: Date.now() + 3600000, // 1 hour expiration
    });

    await newOTPVerification.save();

    // Send email
    await transporter.sendMail(mailOptions);

    console.log("Verification email sent");
  } catch (error) {
    console.error("Error sending verification email:", error);
    res.json({
      status: "FAILED",
      message: "Failed to send verification email",
    });
  }
};

// Send credential email with plain password
const sendCredentialEmail = async ({ email, password, role }, res) => {
  try {
    const mailOptions = {
      from: process.env.AUTH_EMAIL,
      to: email,
      subject: "Credentials for Your Account",
      html: `<p><i>Mabuhay!</i></p>
            <p>You have been created an account for the application. Here are your account details for the BAYBAY SALITA System:</p>
            <h3>Email: ${email}</h3>
            <h3>Password: ${password}</h3>
            <h3>Role: ${role}</h3>
            <p>Please use these credentials to log in.</p>
            <p>Thank you.</p>
            <br>
            <p>Best regards,<br>
            <b>Technical Team</b></p>`,
    };

    // Send email
    await transporter.sendMail(mailOptions);

    console.log("Credential email sent");
  } catch (error) {
    console.error("Error sending credential email:", error);
    res.json({
      status: "FAILED",
      message: "Failed to send credential email",
    });
  }
};

const test = (req, res) => {
  res.json("test is working");
};

// Update the data of the Parent base on the _id
const updateParent = async (req, res) => {
  const { UserID } = req.params;

  try {
    // Destructure only the necessary fields from the request body
    const {
      email,
      FirstName,
      LastName,
      Age,
      Birthday,
      Gender,
      Address,
      Status,
      ContactNumber,
    } = req.body;

    // Validate that the email field is present
    if (!email) {
      return res.status(400).json({
        error: "Email is required",
      });
    }

    // Create an object to hold only the updatable fields
    const updateFields = {
      email,
      FirstName,
      LastName,
      Age,
      Birthday,
      Gender,
      Address,
      Status,
      ContactNumber,
    };

    // Remove any fields that are undefined (optional)
    Object.keys(updateFields).forEach(
      (key) => updateFields[key] === undefined && delete updateFields[key]
    );

    // Update the admin information
    const adminUpdate = await ParentModel.findOneAndUpdate(
      { UserID },
      updateFields,
      {
        new: true, // Return the updated document
        runValidators: true, // Validate the update against the schema
      }
    );

    // Update the user information
    const userUpdate = await User.findOneAndUpdate({ UserID }, updateFields, {
      new: true,
      runValidators: true,
    });

    // Check if either update was successful
    if (!adminUpdate && !userUpdate) {
      return res.status(404).json({
        message: "No account found with this ID",
      });
    }

    // Return the updated admin and user documents
    return res.status(200).json({
      admin: adminUpdate,
      user: userUpdate,
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

//ADD STUDENT MODULE
const addStudent = async (req, res) => {
  // Helper function to check if a value is a valid number
  function isNumber(input) {
    return !isNaN(input) !== ""; // Ensure input isn't empty and is a valid number
  }

  // Consolidated error response
  const errorResponse = (message) => {
    return res.json({ error: message });
  };

  try {
    const {
      LRN,
      FirstName,
      LastName,
      Age,
      Level,
      Section,
      Birthday,
      Address,
      MotherTongue,
      Gender,
      ContactNumber,
    } = req.body;

    // Validate LRN
    if (!LRN) {
      return errorResponse("LRN is required.");
    }
    if (!isNumber(LRN)) {
      return errorResponse("Invalid LRN. It must be numeric.");
    }
    if (LRN.length !== 12) {
      return errorResponse("LRN must be exactly 12 digits long.");
    }

    // Check if the LRN is already in use
    const exist = await StudentModel.findOne({ LRN });
    if (exist) {
      return errorResponse("LRN is already taken.");
    }

    // Validate required fields
    if (!FirstName) return errorResponse("First Name is required.");
    if (!LastName) return errorResponse("Last Name is required.");
    if (!Section) return errorResponse("Section is required.");
    if (!Birthday) return errorResponse("Birthday is required.");
    if (!Address) return errorResponse("Address is required.");
    if (!MotherTongue) return errorResponse("Mother Tongue is required.");
    if (!Gender) return errorResponse("Gender is required.");

    // Validate Age (optional, depending on your requirements)
    if (!Age || !isNumber(Age) || Age < 4 || Age > 100) {
      return errorResponse(
        "Invalid Age. It must be a number between 7 and 100."
      );
    }

    // Create student in the database
    const student = await StudentModel.create({
      LRN,
      FirstName,
      LastName,
      Age,
      Level,
      Section,
      Birthday,
      Address,
      MotherTongue,
      Gender,
      ContactNumber,
      role: "student",
    });

    return res.status(201).json(student); // 201: Created successfully
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" }); // 500: Server error
  }
};

// Gets the whole data of the Students
const getStudents = (req, res) => {
  StudentModel.find()
    .then((students) => res.json(students))
    .catch((err) => res.status(500).json({ error: err.message }));
};

// Gets the data of the Student base on the _id
const getStudent = async (req, res) => {
  if (!mongoose.Types.ObjectId.isValid(req.params.id)) {
    return res.status(400).json({
      message: "Invalid ID format",
    });
  }

  try {
    const student = await StudentModel.findById(req.params.id);
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

// Delete the data of the Student base on the _id
const deleteStudent = async (req, res) => {
  const { id } = req.params;

  if (!mongoose.Types.ObjectId.isValid(id)) {
    return res.status(400).json({
      message: "Invalid ID format",
    });
  }

  try {
    const student = await StudentModel.findByIdAndDelete({ _id: id });
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

// Update the data of the Student base on the _id
const updateStudent = async (req, res) => {
  const { id } = req.params;

  if (!mongoose.Types.ObjectId.isValid(id)) {
    return res.status(400).json({
      message: "Invalid ID format",
    });
  }

  function isNumber(input) {
    return !isNaN(input);
  }

  try {
    const { LRN } = req.body;

    // Check if name is entered
    if (!LRN) {
      return res.json({
        error: "LRN is required",
      });
    }
    if (!isNumber(LRN)) {
      return res.json({ error: "Invalid LRN inputted" });
    }
    if (!LRN.length == 12) {
      return res.json({ error: "LRN must 12 numbers long" });
    }

    const student = await StudentModel.findByIdAndUpdate(
      { _id: id },
      { ...req.body }
    ); // Update credentials in database

    if (!student) {
      return res.status(404).json({
        message: "No account found",
      });
    }

    return res.json(student);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" }); // Add proper error response
  }
};

const getUsers = (req, res) => {
  User.find()
    .then((users) => res.json(users))
    .catch((err) => res.json(err));
};

const getUser = async (req, res) => {
  if (!mongoose.Types.ObjectId.isValid(req.params.id)) {
    return res.status(400).json({
      message: "Invalid ID format",
    });
  }

  try {
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({
        message: "No account found",
      });
    }
    res.status(200).json(user);
  } catch (error) {
    res.status(500).json({
      message: error.message,
    });
  }
};

const getUserID = async (req, res) => {
  const { UserID } = req.params;

  try {
    const user = await User.findOne({ UserID });
    if (!user) {
      return res.json({
        message: "No account found",
      });
    }
    res.status(200).json(user);
  } catch (error) {
    res.status(500).json({
      message: error.message,
    });
  }
};

const getAdmin = async (req, res) => {
  const { UserID } = req.params; // Extract UserID from route parameters

  try {
    // Find the Admin by UserID
    const user = await Admin.findOne({ UserID });

    // If no teacher is found, return a 404 status
    if (!user) {
      return res.json({
        message: "No account found",
      });
    }

    // Return the found Admin
    res.json(user);
  } catch (error) {
    // Handle any other errors
    res.json({
      message: error.message,
    });
  }
};

// Add User function
const addUser = async (req, res) => {
  function validatePassword(password) {
    const minLength = 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasDigit = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (password.length < minLength) {
      return "Password must be at least 8 characters long.";
    }
    if (!hasUpperCase) {
      return "Password must contain at least one uppercase letter.";
    }
    if (!hasLowerCase) {
      return "Password must contain at least one lowercase letter.";
    }
    if (!hasDigit) {
      return "Password must contain at least one digit.";
    }
    if (!hasSpecialChar) {
      return "Password must contain at least one special character.";
    }

    return null; // No error
  }

  try {
    const { UserID, email, password, role } = req.body;

    // Validate email and required fields
    if (!email) {
      return res.json({ error: "Email is required" });
    }

    if (!password) {
      return res.json({ error: "Password is required" });
    }

    if (!role) {
      return res.json({ error: "Role is required" });
    }

    // Check if email already exists
    const exist = await User.findOne({ email });
    if (exist) {
      return res.json({ error: "Email is already taken" });
    }

    // Validate password
    const passwordError = validatePassword(password);
    if (passwordError) {
      return res.json({ error: passwordError });
    }

    // Hash the password
    const hashedPassword = await hashPassword(password);

    // Create and save new user
    const user = new User({
      UserID,
      email,
      password: hashedPassword,
      role,
      verified: false,
    });

    await user.save();

    // If role is Teacher, save to TeacherModel
    if (role === "Teacher") {
      const teacher = new Teacher({
        UserID: user.UserID,
        email: user.email,
        createdAt: new Date(),
      });

      await teacher.save();
      console.log("Teacher saved successfully");
    }

    // Send plain password via credentials email
    await sendCredentialEmail({ email, password, role }, res);

    res.json({
      status: "SUCCESS",
      message: "User created successfully",
    });
  } catch (error) {
    console.error("Error adding user:", error);
    res.status(500).json({ error: "Server error" });
  }
};

// Update the data of the Student base on the _id
const updateUser = async (req, res) => {
  const { UserID } = req.params;

  function validatePassword(password) {
    const minLength = 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasDigit = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (password.length < minLength) {
      return "Password must be at least 8 characters long.";
    }
    if (!hasUpperCase) {
      return "Password must contain at least one uppercase letter.";
    }
    if (!hasLowerCase) {
      return "Password must contain at least one lowercase letter.";
    }
    if (!hasDigit) {
      return "Password must contain at least one digit.";
    }
    if (!hasSpecialChar) {
      return "Password must contain at least one special character.";
    }

    return null; // No error
  }

  try {
    const { FirstName, LastName, email, password, role } = req.body;

    if (!FirstName) {
      return res.json({
        error: "Password is required",
      });
    }

    if (!LastName) {
      return res.json({
        error: "Password is required",
      });
    }

    if (!password) {
      return res.json({
        error: "Password is required",
      });
    }

    const passwordError = validatePassword(password);
    if (passwordError) {
      return res.json({ error: passwordError });
    }

    if (!role) {
      return res.json({
        error: "Role is required",
      });
    }

    const hashedPassword = await hashPassword(password);

    const user = await User.findOneAndUpdate(
      { UserID: UserID },
      {
        FirstName,
        LastName,
        email,
        password: hashedPassword,
        role,
      }
    ); // Update credentials in database

    if (!user) {
      return res.status(404).json({
        message: "No account found",
      });
    }

    return res.json(user);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" }); // Add proper error response
  }
};

const deleteUser = async (req, res) => {
  const { email } = req.params;

  try {
    // Step 1: Find and delete the user by email
    const user = await User.findOneAndDelete({ email });

    if (!user) {
      return res.json({
        message: "No User found with this email",
      });
    }

    // Step 2: Find and delete the associated teacher and parent records if they exist
    const teacherDeleted = await Teacher.findOneAndDelete({ email });
    const parentDeleted = await ParentModel.findOneAndDelete({ email });

    // Step 6: Respond with success message after deletion
    res.json({
      message: "User and associated data deleted successfully",
      user,
      teacherDeleted, // Indicates if a teacher was deleted
      parentDeleted, // Indicates if a parent was deleted
    });
  } catch (error) {
    // Error handling
    res.json({
      message: error.message,
    });
  }
};

//Endpoint getProfile with get res
//Verifying the user
const getProfile = (req, res) => {
  const { token } = req.cookies;
  if (token) {
    jwt.verify(token, process.env.JWT_SECRET, {}, (err, user) => {
      if (err) throw err;
      res.json(user);
    });
  } else {
    res.json(null);
  }
};

module.exports = {
  test, //okay
  addStudent, //okay
  getStudents, //okay
  getStudent, //okay
  deleteStudent,
  updateStudent,
  updateParent,
  addUser,
  getUsers,
  getUserID,
  getAdmin,
  getUser,
  updateUser,
  deleteUser,
  getProfile,
};
