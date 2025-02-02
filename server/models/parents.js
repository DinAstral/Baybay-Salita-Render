const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const parentsSchema = new Schema(
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
    Age: String,
    Birthday: String,
    Address: String,
    Status: String,
    Gender: String,
    ContactNumber: String,
    Student: Object,
    role: String,
    Picture: String,
    verified: Boolean,
  },
  { timestamps: true }
);

const ParentModel = mongoose.model("Parents", parentsSchema);

module.exports = ParentModel;
