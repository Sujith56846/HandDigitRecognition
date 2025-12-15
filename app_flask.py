import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# Model Classes (same as in digit_recognition.py)
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First convolutional block
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        
        # Second convolutional block
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        
        # Third convolutional block
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Global variables for model
model = None
model_type = None


def load_model():
    """Load the pre-trained model (CNN preferred, ANN as fallback)"""
    global model, model_type
    
    cnn_model_path = 'digit_recognition_cnn_model.pth'
    ann_model_path = 'digit_recognition_ann_model.pth'
    
    # Try to load CNN model first (best performance)
    if os.path.exists(cnn_model_path):
        model = CNN()
        model.load_state_dict(torch.load(cnn_model_path, map_location='cpu'))
        model.eval()
        model_type = "CNN"
        print(f"CNN Model loaded successfully from {cnn_model_path}")
        return True
    
    # Fallback to ANN model
    elif os.path.exists(ann_model_path):
        model = ANN()
        model.load_state_dict(torch.load(ann_model_path, map_location='cpu'))
        model.eval()
        model_type = "ANN"
        print(f"ANN Model loaded successfully from {ann_model_path}")
        return True
    
    # No model found
    else:
        print("No trained model found!")
        print("Please run 'train_model.py' first to train and save the models.")
        return False


def process_canvas_image(image_data):
    """Process the canvas image data for model prediction"""
    try:
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize to 28x28 (MNIST size)
        image_resized = cv2.resize(image_array, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [-1, 1] (same as training)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - 0.5) / 0.5
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(image_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def predict_digit(image_tensor):
    """Predict the digit from processed image tensor"""
    if model is None or image_tensor is None:
        return None, 0.0
    
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        return prediction.item(), confidence.item() * 100
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', model_type=model_type)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        image_tensor = process_canvas_image(data['image'])
        
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        prediction, confidence = predict_digit(image_tensor)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': round(confidence, 1),
            'model_type': model_type
        })
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': model_type
    })


if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        print(f"\n🚀 Hand Digit Recognition Web App")
        print(f"📊 Model Type: {model_type}")
        print(f"🌐 Starting Flask server...")
        print(f"📱 Access locally: http://localhost:5000")
        print(f"🌍 Access globally: http://0.0.0.0:5000")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load model. Please train the model first by running 'train_model.py'")