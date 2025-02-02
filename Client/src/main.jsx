import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter as Router } from "react-router-dom";
import App from "./App.jsx";
import "./index.css";

const fontLink = document.createElement("link");
fontLink.rel = "preload";
fontLink.href =
  "https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap";
fontLink.as = "style";
fontLink.onload = () => (fontLink.rel = "stylesheet");
document.head.appendChild(fontLink);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Router>
      <App />
    </Router>
  </StrictMode>
);
