const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const teacherSchema = new Schema(
  {
    UserID: {
      type: String,
      unique: true,
    },
    email: {
      type: String,
      unique: true,
    },
    FirstName: String,
    LastName: String,
    Section: String,
    Department: String,
    Age: String,
    Birthday: Object,
    Address: String,
    Status: String,
    Gender: String,
    ContactNumber: String,
    Picture: String,
    role: String,
  },
  { timestamps: true }
);

const TeacherModel = mongoose.model("Teachers", teacherSchema);

module.exports = TeacherModel;
