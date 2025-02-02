const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const studentSchema = new Schema(
  {
    LRN: {
      type: String,
      unique: true,
    },
    FirstName: String,
    LastName: String,
    Age: String,
    Section: String,
    Birthday: String,
    Address: String,
    MotherTongue: String,
    ContactNumber: String,
    Gender: String,
    Picture: String,
    status: String,
    role: String,
  },
  { timestamps: true }
);

const StudentModel = mongoose.model("Students", studentSchema);

module.exports = StudentModel;
