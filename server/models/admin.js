const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const adminSchema = new Schema(
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

const AdminModel = mongoose.model("Admin", adminSchema);

module.exports = AdminModel;
