# MNIST Digit Recognizer

This project is a web-based application that allows users to draw digits on a canvas and uses a Convolutional Neural Network (CNN) model to recognize and predict the drawn digits. The model is trained on the MNIST dataset and achieves high accuracy in digit recognition.

## Features
- Draw digits on an interactive canvas.
- Submit the drawing to receive a prediction of the digit along with the confidence level.
- Clear the canvas to draw a new digit.
- Uses a Flask backend to serve the model and handle predictions.

## Technologies Used
- Python
- Flask
- PyTorch
- torchvision
- Pillow
- HTML/CSS/JavaScript

## Project Structure
├── app.py
├── data_loading.py
├── models.py
├── templates
│   └── index.html
├── static
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
## Setup Instructions

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mnist-digit-recognizer.git
    cd mnist-digit-recognizer
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the MNIST dataset:
    ```bash
    python data_loading.py
    ```

## Usage

1. **Train the Model (if needed):**
   If you haven't already trained the model or want to retrain it, you can use the training script. Ensure `data_loading.py` is set up to load the MNIST dataset correctly.
   ```bash
   python train_model.py
   ```

2. **Run the Flask Application:**
    ```bash
    python app.py
    ```

3. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the digit recognizer.

## File Descriptions

- **app.py:** The main Flask application file that handles web requests, processes images, and serves predictions.
- **data_loading.py:** Contains functions to download and load the MNIST dataset.
- **models.py:** Defines the CNN model architecture for digit recognition.
- **templates/index.html:** HTML template for the web interface.
- **static/style.css:** CSS file for styling the web interface.
- **static/script.js:** JavaScript file for handling the drawing canvas and user interactions.

## How It Works

1. **Drawing Canvas:** Users can draw a digit on the provided canvas.
2. **Submit Button:** When the user clicks "Submit", the canvas content is sent to the Flask backend.
3. **Image Preprocessing:** The backend preprocesses the image (inverts colors, resizes, normalizes).
4. **Model Prediction:** The preprocessed image is fed into the CNN model to get the predicted digit and confidence level.
5. **Display Result:** The prediction and confidence are displayed to the user.


