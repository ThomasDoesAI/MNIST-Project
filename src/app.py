from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps
import io
import base64
import re
import numpy
from models import MNISTModel

app = Flask(__name__)

model = MNISTModel()
model.load_state_dict(torch.load('models/mnist_model.pth', map_location=torch.device('cpu')))
model.eval()


# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def preprocess_image(image_data):
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.convert('L')
    image = PIL.ImageOps.invert(image)
    image = transform(image).unsqueeze(0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    #if 'image' not in request.json:
    #    return jsonify({'error': 'No file provided'})
    data = request.get_json()
    data = data.get('image', None)
    image = preprocess_image(data)
    
    with torch.no_grad():
        output = model(image)
        print("Model Output: ",output)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()
        print({'digit': predicted.item(), 'confidence': confidence})
    return jsonify({'digit': predicted.item(), 'confidence': confidence})

if __name__ == "__main__":
    app.run(debug=True)