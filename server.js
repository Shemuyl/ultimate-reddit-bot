const express = require("express");
const path = require("path");
const app = express();

// Use Railway's dynamic port or default 3000 locally
const PORT = process.env.PORT || 3000;

// Serve static files (CSS, JS) from "public" folder
app.use(express.static(path.join(__dirname, "public")));

// Serve HTML files from "views" folder (if you have EJS)
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "ejs");

// Example home route
app.get("/", (req, res) => {
  res.render("index"); // Make sure you have views/index.ejs
});

// Optional test route for raw HTML
app.get("/hello", (req, res) => {
  res.send("Hello from Node.js + Express 🚀");
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
