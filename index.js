const express = require('express');
const app = express();
const port = 3000;

// Step 1: Define what happens when someone goes to "/"
app.get('/', (req, res) => {
  res.send('Hello World! Your server is working 🚀');
});

// Step 2: Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
