const mongoose = require('mongoose'); //(npm i mongoose)
const {Schema} = mongoose

const tokenSchema = new Schema ({
      userId: String,
      otp: String,
      createdAt: Date,
      expiresAt: Date,

}, {timestamps: true})


const UserOTPVerification = mongoose.model('token', tokenSchema);


module.exports = UserOTPVerification;