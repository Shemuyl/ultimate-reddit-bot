const express = require('express');
const path = require('path');
const app = express();

// Railway will set the PORT for you, fallback to 3000 for local dev
const PORT = process.env.PORT || 3000;

// Serve static files (CSS, JS, Images)
app.use(express.static(path.join(__dirname, 'public')));

// Serve index.html for root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Catch-all route for other paths (useful if you have a SPA like React/Vue)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});

