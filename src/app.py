from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps
import io
import base64
import re
from models import MNISTModel

app = Flask(__name__)

# Load the trained model
model = MNISTModel()
model.load_state_dict(torch.load('models/mnist_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image tensor
])

def preprocess_image(image_data):
    """
    Preprocess the image data by decoding from base64, inverting colors, and applying transformations.
    """
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.convert('L')  # Convert to grayscale
    image = PIL.ImageOps.invert(image)  # Invert colors to match MNIST format
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    return image

@app.route('/')
def index():
    """
    Render the main page with the drawing canvas.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the digit from the drawn image and return the result as JSON.
    """
    data = request.get_json()
    image_data = data.get('image', None)
    
    if image_data is None:
        return jsonify({'error': 'No image provided'}), 400
    
    image = preprocess_image(image_data)  # Preprocess the image

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(image)  # Get the model output
        _, predicted = torch.max(output, 1)  # Get the predicted digit
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()  # Calculate confidence

    return jsonify({'digit': predicted.item(), 'confidence': confidence})  # Return the prediction and confidence

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
