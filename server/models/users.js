const mongoose = require("mongoose"); //(npm i mongoose)
const { Schema } = mongoose;

const userSchema = new Schema(
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
    password: String,
    role: String,
    Picture: String,
    verified: Boolean,
  },
  { timestamps: true }
);

const UserModel = mongoose.model("Users", userSchema);

module.exports = UserModel;
