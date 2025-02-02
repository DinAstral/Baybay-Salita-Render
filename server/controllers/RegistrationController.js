const mongoose = require("mongoose");
const User = require("../models/users");
const Admin = require("../models/admin");
const Teacher = require("../models/teachers");
const Parent = require("../models/parents");
const Student = require("../models/student");
const { hashPassword, comparePassword } = require("../helpers/auth");
const jwt = require("jsonwebtoken");
const UserOTPVerification = require("../models/UserOTPVerification");
const nodemailer = require("nodemailer");
const bcrypt = require("bcryptjs");

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

function generateRandomCodeAdmin(length) {
  const characters = "0123456789";
  let result = "adminID_";
  const charactersLength = characters.length;
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

function generateRandomCodeParent(length) {
  const characters = "0123456789";
  let result = "parentID_";
  const charactersLength = characters.length;
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

function generateRandomCodeTeacher(length) {
  const characters = "0123456789";
  let result = "teacherID_";
  const charactersLength = characters.length;
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

let transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  auth: {
    user: process.env.AUTH_EMAIL,
    pass: process.env.AUTH_PASS,
  },
});

const sendCredentialEmail = async ({ email, password, role }, res) => {
  try {
    const mailOptions = {
      from: process.env.AUTH_EMAIL,
      to: email,
      subject: "Credentials for Your Account",
      html: `<p><i>Mabuhay!</i></p>
              <p>You have been created an account for the application. Here's the details of your account for the BAYBAY SALITA System:</p>
              <h3>Email: ${email}</h3>
              <h3>Password: ${password}</h3>
              <h3>Role: ${role}</h3>
              <p>To verify your account please use these credentials to login and input the verification code that will be given to you.</p>
              <p>If you need any assistance, please don't hesitate to contact our team.</p>
              <p>Thank you.</p>
              <br>
              <p>Best regards,<br>
              <b>Technical Team</b></p>`,
    };

    transporter.sendMail(mailOptions);
    console.log("Success");
    res.json({
      status: "SUCCESS",
      message: "Credential email sent",
      data: { email, password },
    });
  } catch (error) {
    res.json({
      status: "FAILED",
      message: error.message,
    });
  }
};

const sendForgotPasswordEmail = async ({ email, link }, res) => {
  try {
    const mailOptions = {
      from: process.env.AUTH_EMAIL,
      to: email,
      subject: "Reset your account",
      html: `<p><i>Mabuhay!</i></p>
             <p><b><i>This is BaybaySalita Admin!</i></b></p>
              <p>To reset you account password please follow this link! This link is only valid for 30 minutes before it expires.</p>
              <h4>Link: <a href="${link}">Reset Password</a> </h4>
              <p>If you need any assistance, please don't hesitate to contact our team.</p>
              <p>Thank you.</p>
              <br>
              <p>Best regards,<br>
              <b>Technical Team</b></p>`,
    };

    transporter.sendMail(mailOptions);
    console.log("Success");
    res.json({
      status: "SUCCESS",
      message: "Credential email sent",
      data: { email, password },
    });
  } catch (error) {
    res.json({
      status: "FAILED",
      message: error.message,
    });
  }
};

const sendVerificationEmail = async ({ UserID, email }, res) => {
  try {
    const otp = `${Math.floor(100000 + Math.random() * 900000)}`;

    const mailOptions = {
      from: process.env.AUTH_EMAIL,
      to: email,
      subject: "Verify Your Email",
      html: `<p><i>Mabuhay!</i></p>
              <p>We received a registration request to your account. To verify your account, please use the following One-Time Password (OTP) code.</p>
              <h3>OTP: ${otp}</h3>
              <p>This code will <b>expire in 1 hour.</b></p>
              <p>Please enter this code on the OTP verification page to complete the process. If you didn't initiate this request, please disregard this email.</p>
              <p>If you need any assistance, please don't hesitate to contact our team.</p>
              <p>Thank you.</p>
              <br>
              <p>Best regards,<br>
              <b>BaybaySalita</b>`,
    };

    const saltrounds = 10;

    const hashedOTP = await bcrypt.hash(otp, saltrounds);
    const newOTPVerification = await new UserOTPVerification({
      userId: UserID,
      otp: hashedOTP,
      createdAt: Date.now(),
      expiresAt: Date.now() + 3600000,
    });
    await newOTPVerification.save();
    transporter.sendMail(mailOptions);
    console.log("Success");
    res.json({
      status: "PENDING",
      message: "Verification otp email sent",
      data: {
        userId: UserID,
        email,
      },
    });
  } catch (error) {
    res.json({
      status: "FAILED",
      message: error.message,
    });
  }
};

const verifyOTP = async (req, res) => {
  try {
    const { userId, otp } = req.body;

    if (!userId || !otp) {
      return res.json({ error: "Empty OTP details are not allowed" });
    }

    // Fetch the OTP verification records for the given userId
    const UserOTPVerificationRecords = await UserOTPVerification.find({
      userId,
    });
    if (UserOTPVerificationRecords.length === 0) {
      return res.json({
        error:
          "Account record doesn't exist or has been verified already. Please sign up or log in.",
      });
    }

    // Extract the first record (assuming there could be multiple records)
    const { expiresAt, otp: hashedOTP } = UserOTPVerificationRecords[0];

    // Check if the OTP has expired
    if (expiresAt < Date.now()) {
      await UserOTPVerification.deleteMany({ userId });
      return res.json({ error: "Code is expired. Please request again." });
    }

    // Ensure the hashedOTP is defined before comparing
    if (!hashedOTP) {
      return res.json({ error: "he" });
    }

    // Compare the provided OTP with the stored hashed OTP
    const validOTP = await bcrypt.compare(otp, hashedOTP);

    if (!validOTP) {
      return res.json({ error: "Invalid code. Please check your email." });
    }

    // Mark the user as verified
    await User.updateOne({ UserID: userId }, { verified: true });
    await Parent.updateOne({ UserID: userId }, { verified: true });
    await Teacher.updateOne({ UserID: userId }, { verified: true });
    await UserOTPVerification.deleteMany({ userId });

    res.json({
      status: "VERIFIED",
      message: "User email verified successfully.",
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const registerTeacher = async (req, res) => {
  const TeacherID = generateRandomCodeTeacher(6);

  try {
    const {
      FirstName,
      LastName,
      Section,
      Department,
      Birthday,
      Age,
      Address,
      Status,
      Gender,
      ContactNumber,
      email,
      password,
    } = req.body;

    const exist = await User.findOne({ email });
    if (exist) {
      return res.json({ error: "Email has already been taken." });
    }

    const hashedPassword = await hashPassword(password); // Hash the password

    // Create teacher and User records
    const teacher = new Teacher({
      UserID: TeacherID,
      FirstName,
      LastName,
      Section,
      Department,
      Birthday,
      Age,
      Gender,
      Address,
      Status,
      ContactNumber,
      email,
      password: hashedPassword,
      role: "Teacher",
      verified: false,
    });

    const user = new User({
      UserID: TeacherID,
      FirstName,
      LastName,
      email,
      role: "Teacher",
      password: hashedPassword,
      verified: false,
    });

    // Save both admin and user records
    await teacher.save();
    await user
      .save()
      .then((result) => {
        sendVerificationEmail(result, res);
      })
      .catch((err) => {
        console.log(err);
        res.json({
          status: "FAILED",
          message: "An Error occurred while saving your account!",
        });
      });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const uploadFile = async (req, res) => {
  try {
    if (!req.file) {
      return res.json({ error: "No file uploaded" });
    }

    const { TeacherID } = req.body;

    if (!TeacherID) {
      return res.json({ error: "TeacherID is required" });
    }

    // Find the teacher and update the Picture field
    const teacher = await Teacher.findOneAndUpdate(
      { UserID: TeacherID },
      { Picture: req.file.filename },
      { new: true }
    );

    if (!teacher) {
      return res.status(404).json({ error: "Teacher not found" });
    }

    return res.json({
      status: "SUCCESS",
      message: "File uploaded and associated with teacher successfully",
      teacher,
    });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const registerParent = async (req, res) => {
  const ParentID = generateRandomCodeParent(6);

  try {
    const {
      FirstName,
      LastName,
      LRN,
      Age,
      StudentFirstName,
      StudentLastName,
      Birthday,
      StudentBirthday,
      StudentAge,
      Address,
      Status,
      Gender,
      StudentGender,
      MotherTongue,
      ContactNumber,
      email,
      password,
    } = req.body;

    const exist = await User.findOne({ email });
    if (exist) {
      return res.json({ error: "Email has already been taken." });
    }

    const hashedPassword = await hashPassword(password); // Hash the password

    let student = [
      {
        FirstName: StudentFirstName,
        LastName: StudentLastName,
        LRN: LRN,
        Age: StudentAge,
        Birthday: StudentBirthday,
        Gender: StudentGender,
        MotherTongue: MotherTongue,
      },
    ];

    // Create Parent and User records
    const parent = new Parent({
      UserID: ParentID,
      FirstName,
      LastName,
      Birthday,
      Age,
      Address,
      Status,
      Gender,
      ContactNumber,
      email,
      password: hashedPassword,
      Student: student,
      role: "Parent",
      verified: false,
    });

    const user = new User({
      UserID: ParentID,
      FirstName,
      LastName,
      email,
      role: "Parent",
      password: hashedPassword,
      verified: false,
    });

    // Save both teacher and user records
    await parent.save();
    await user
      .save()
      .then((result) => {
        sendVerificationEmail(result, res);
      })
      .catch((err) => {
        console.log(err);
        res.json({
          status: "FAILED",
          message: "An Error occurred while saving your account!",
        });
      });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const registerAdmin = async (req, res) => {
  const adminID = generateRandomCodeAdmin(6);

  function isNumber(input) {
    return !isNaN(input);
  }

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
    const {
      FirstName,
      LastName,
      Birthday,
      Address,
      Status,
      Gender,
      ContactNumber,
      email,
      password,
    } = req.body;

    if (!FirstName) {
      return res.json({ error: "First Name is required" });
    }
    if (!LastName) {
      return res.json({ error: "Last Name is required" });
    }
    if (!Birthday) {
      return res.json({ error: "Birthday is required" });
    }
    if (!Gender) {
      return res.json({ error: "Gender is required" });
    }
    if (!Address) {
      return res.json({ error: "Address is required" });
    }
    if (!Status) {
      return res.json({ error: "Status is required" });
    }
    if (!ContactNumber) {
      return res.json({ error: "Contact Number is required" });
    }
    if (!isNumber(ContactNumber)) {
      return res.json({ error: "Invalid Contact Number inputted" });
    }
    if (!email) {
      return res.json({ error: "Email is required" });
    }

    const exist = await User.findOne({ email });
    if (exist) {
      return res.json({ error: "Email has already been taken." });
    }

    const passwordError = validatePassword(password);
    if (passwordError) {
      return res.json({ error: passwordError });
    }

    const hashedPassword = await hashPassword(password); // Hash the password

    // Create admin and User records
    const admin = new Admin({
      UserID: adminID,
      FirstName,
      LastName,
      Birthday,
      Gender,
      Address,
      Status,
      ContactNumber,
      email,
      password: hashedPassword,
      role: "Admin",
      verified: false,
    });

    const user = new User({
      UserID: adminID,
      FirstName,
      LastName,
      email,
      role: "Admin",
      password: hashedPassword,
      verified: false,
    });

    // Save both teacher and user records
    await admin.save();
    await user
      .save()
      .then((result) => {
        sendVerificationEmail(result, res);
      })
      .catch((err) => {
        console.log(err);
        res.json({
          status: "FAILED",
          message: "An Error occurred while saving your account!",
        });
      });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    // Check if both email and password are provided
    if (!email && !password) {
      return res.json({ error: "Email and Password are required." });
    }

    // Validate email format
    if (!emailRegex.test(email)) {
      return res.json({ error: "Invalid email format." });
    }

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.json({ error: "Email does not exist. Please try again." });
    }

    // Check if the user's email is verified
    if (!user.verified) {
      sendVerificationEmail({ UserID: user._id, email: user.email });
      return res.json({
        error: "Account is not verified. A verification email has been sent.",
        data: { userId: user._id },
      });
    }

    // Check if the password matches
    const isMatch = await comparePassword(password, user.password);
    if (!isMatch) {
      return res.json({ error: "Email and Password do not match." });
    }

    // Generate JWT token
    jwt.sign(
      { email: user.email },
      process.env.JWT_SECRET,
      {},
      (err, token) => {
        if (err) {
          console.error(err);
          return res.json({ error: "Failed to generate token" });
        }
        // Set the token in an HTTP-only cookie and respond with success
        return res
          .cookie("token", token, { httpOnly: true })
          .json({ success: true, token, role: user.role, user });
      }
    );
  } catch (error) {
    console.error("Error in loginUser:", error);
    return res.json({ error: "Server error occurred" });
  }
};

const forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;

    const userEmail = await User.findOne({ email });
    if (!userEmail) {
      return res.json({
        error: "No account found for this email address.",
      });
    } else {
      const secret = process.env.JWT_SECRET + userEmail.password;
      const token = jwt.sign({ email: userEmail, id: userEmail._id }, secret, {
        expiresIn: "30m",
      });
      const link = `https://baybay-salita-heroku-8c328f3ddd0f.herokuapp.com/api/reset-password/${userEmail._id}/${token}`;
      console.log(link);
      sendForgotPasswordEmail({ email, link }, res);
    }
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const resetPassword = async (req, res) => {
  try {
    const { id, token } = req.params;
    const { password } = req.body;

    const user = await User.findById(id);
    if (!user) {
      return res.json({ error: "No account found for this email address." });
    }

    const passwordError = validatePassword(password);
    if (passwordError) {
      return res.json({ error: passwordError });
    }

    const secret = process.env.JWT_SECRET + user.password;

    jwt.verify(token, secret, async (err, decoded) => {
      if (err) {
        return res.json({ status: "Error with token" });
      } else {
        try {
          const hashedPassword = await hashPassword(password);
          user.password = hashedPassword;
          await user.save();
          return res.json({
            status: "Success",
            message: "Password updated successfully.",
          });
        } catch (err) {
          console.error(err);
          return res.status(500).json({
            status: "FAILED",
            message: "An error occurred while saving your account!",
          });
        }
      }
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

const logout = (req, res) => {
  res.clearCookie("token", { path: "/" }); // Adjust the path as necessary
  res.status(200).json({ message: "Logged out successfully." });
};

const { profileUpload } = require("../middleware/multer");
const { cloudinaryUploaderProfile } = require("../hooks/cloudinary");
const multer = require("multer");

const profileUpdate = async (req, res) => {
  profileUpload(req, res, async function (err) {
    if (err instanceof multer.MulterError) {
      return res.json({ error: `Multer Error: ${err.message}` });
    } else if (err) {
      return res.json({ error: `Error: ${err.message}` });
    }

    try {
      // Extracting Type, Role, and ID from request body
      const { role, UserID, LRN } = req.body;
      if (!role) {
        return res.json({ error: "Can't update your profile." });
      }

      // Uploading to Cloudinary
      const uploadResponse = await cloudinaryUploaderProfile(req, res);
      const profileFile = uploadResponse.uploadProfile.secure_url;

      // Finding and updating the existing record based on role
      let updatedRecord;
      switch (role) {
        case "Admin":
          updatedRecord = await Admin.findOneAndUpdate(
            { UserID: UserID }, // Filter: Find Admin with this UserID
            { Picture: profileFile }, // Update: Set new profile picture URL
            { new: true } // Return the updated document
          );
          break;
        case "Teacher":
          updatedRecord = await Teacher.findOneAndUpdate(
            { UserID: UserID }, // Filter: Find Teacher with this UserID
            { Picture: profileFile }, // Update: Set new profile picture URL
            { new: true } // Return the updated document
          );
          break;
        case "Parent":
          updatedRecord = await Parent.findOneAndUpdate(
            { UserID: UserID }, // Filter: Find Parent with this UserID
            { Picture: profileFile }, // Update: Set new profile picture URL
            { new: true } // Return the updated document
          );
          break;
        case "student":
          updatedRecord = await Student.findOneAndUpdate(
            { LRN: LRN }, // Filter: Find Student with this LRN
            { Picture: profileFile }, // Update: Set new profile picture URL
            { new: true } // Return the updated document
          );
          break;
        default:
          return res.json({ error: "Invalid role" });
      }

      // Only update the User collection if the role is not "student"
      if (updatedRecord && role !== "student") {
        await User.findOneAndUpdate(
          { UserID: UserID }, // Filter: Find User with this UserID
          { Picture: profileFile }, // Update: Set new profile picture URL
          { new: true } // Return the updated document
        );
      }

      if (!updatedRecord) {
        return res.json({ error: "User not found or update failed." });
      }

      return res.json({
        message: "Profile picture successfully updated.",
        updatedProfileUrl: profileFile,
      });
    } catch (error) {
      console.error(error);
      return res.status(500).json({ error: "Server error" });
    }
  });
};

const updateAdmin = async (req, res) => {
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
        error: "Email is required.",
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
    const adminUpdate = await Admin.findOneAndUpdate({ UserID }, updateFields, {
      new: true, // Return the updated document
      runValidators: true, // Validate the update against the schema
    });

    // Update the user information
    const userUpdate = await User.findOneAndUpdate({ UserID }, updateFields, {
      new: true,
      runValidators: true,
    });

    // Check if either update was successful
    if (!adminUpdate && !userUpdate) {
      return res.status(404).json({
        message: "No account found with this ID.",
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

module.exports = {
  registerAdmin,
  registerParent,
  registerTeacher,
  uploadFile,
  verifyOTP,
  loginUser,
  forgotPassword,
  resetPassword,
  logout,
  profileUpdate,
  updateAdmin,
};
