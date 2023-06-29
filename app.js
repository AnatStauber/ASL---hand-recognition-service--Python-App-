const express = require("express");
const path = require("path");
const http = require("http");
const cors = require("cors");
const {routesInit} = require("./routes/config_routes")
// require("./db/mongoconnect");
// require("./utils/imageCompare");
const { spawn } = require('child_process');

// Run the Python script
const pythonProcess = spawn('python', ['./utils/trainCNN.py']);

const app = express();

// נותן גישה לכל הדומיינים לגשת לשרת שלנו
app.use(cors());
// כדי שנוכל לקבל באדי
app.use(express.json());
// הגדרת תקיית הפאבליק כתקייה ראשית
app.use(express.static(path.join(__dirname,"public")))

routesInit(app);

const server = http.createServer(app);

let port = process.env.PORT || 3000
server.listen(port);

// Listen for Python script output
pythonProcess.stdout.on('data', (data) => {
    console.log(`Python script output: ${data}`);
  });
  
  // Listen for Python script exit
  pythonProcess.on('close', (code) => {
    console.log(`Python script exited with code ${code}`);
  });
  
  // Handle any errors
  pythonProcess.on('error', (err) => {
    console.error(`Error executing Python script: ${err}`);
  });