const express = require("express");
require("dotenv").config();
const cors = require("cors");
const mongoose = require("mongoose");
const cookieParser = require("cookie-parser");
const path = require("path");
const cloudinary = require("cloudinary").v2;
const compression = require("compression");
const helmet = require("helmet"); // Import Helmet
const crypto = require("crypto");

const app = express();

// Enable compression for better performance
app.use(compression());

app.use((req, res, next) => {
  res.locals.nonce = crypto.randomBytes(16).toString("hex");
  next();
});

// Apply security headers using Helmet
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"], // Allow self-hosted content
        scriptSrc: ["'self'", "'unsafe-inline'"], // Allow inline/eval for certain cases
        styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"], // Allow Google Fonts for NextUI
        fontSrc: ["'self'", "https://fonts.gstatic.com"], // Allow fonts from Google Fonts
        imgSrc: ["'self'", "data:", "https://res.cloudinary.com"], // Allow images from Cloudinary
        connectSrc: ["'self'", "https://api.web3forms.com"], // Allow connections to APIs
        frameSrc: ["'self'"], // For iframes (optional, if used)
        childSrc: ["'self'"], // For embedded content (optional)
        mediaSrc: ["'self'", "data:", "https://res.cloudinary.com"], // For media content
        objectSrc: ["'none'"], // Disallow embedding of objects (security)
      },
    },
    referrerPolicy: { policy: "no-referrer" },
    frameguard: { action: "deny" },
    hsts: { maxAge: 63072000, includeSubDomains: true },
    xssFilter: true,
    noSniff: true,
    permissionsPolicy: {
      features: {
        geolocation: ["'none'"],
        camera: ["'none'"],
        microphone: ["'none'"],
        payment: ["'none'"],
        usb: ["'none'"],
        accelerometer: ["'none'"],
      },
    },
  })
);

app.use((req, res, next) => {
  res.setHeader(
    "Permissions-Policy",
    "geolocation=(self), camera=(), microphone=()"
  ); // Adjust as needed
  next();
});

// Database connection
const uri = process.env.MONGODB_URI;
mongoose
  .connect(uri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Database is Connected"))
  .catch((err) => console.error("Database not connected", err));

// Cloudinary Setup
cloudinary.config({
  cloud_name: "dvcqnbkwb",
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// Middleware for parsing JSON and URL-encoded data
app.use(express.json({ limit: "100mb" }));
app.use(cookieParser());
app.use(express.urlencoded({ extended: false }));

// Register API routes
app.use("/api", require("./routes/authRoutes"));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send("Something went wrong!");
});

// Serve static files from the client build (React Vite build)
app.use(express.static(path.join(__dirname, "../Client/dist")));

// Catch-all route to serve the React app (if no API route matches)
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "../Client/dist/index.html"), (err) => {
    if (err) {
      res.status(500).send(err);
    }
  });
});

// Start the server on the specified port
const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`Server is running on port ${port}`));
