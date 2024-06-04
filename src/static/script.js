// Get the canvas element and its 2D drawing context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set the initial canvas background to white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

let painting = false; // Flag to track whether the user is drawing

// Function to start drawing
function startPosition(e) {
    painting = true;
    draw(e);
}

// Function to stop drawing
function endPosition() {
    painting = false;
    ctx.beginPath(); // Begin a new path (to stop connecting lines)
}

// Function to draw on the canvas
function draw(e) {
    if (!painting) return; // Only draw if the mouse is pressed

    ctx.lineWidth = 10; // Set the line width
    ctx.lineCap = 'round'; // Set the line cap to round
    ctx.strokeStyle = 'black'; // Set the stroke color to black

    // Draw a line to the current mouse position
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath(); // Begin a new path
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

// Event listeners for mouse actions
canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);

// Function to clear the canvas
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill it with white color
    document.getElementById('result').innerText = ''; // Clear the result text
}

// Function to submit the canvas drawing to the server
async function submitCanvas() {
    const dataURL = canvas.toDataURL(); // Get the data URL of the canvas content

    // Send the data URL to the server for prediction
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: dataURL }),
    });

    // Get the result from the server
    const result = await response.json();
    
    // Display the result
    document.getElementById('result').innerText = 'Number: ' + result.digit + ' Confidence: ' + result.confidence.toFixed(2);
}
