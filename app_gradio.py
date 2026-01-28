"""
Gradio version of Hand Digit Recognition for Hugging Face Spaces
This provides a simpler alternative to the Flask app with native HF support
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2


# CNN Model Definition
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


# Load model
def load_model():
    model = CNN()
    model.load_state_dict(torch.load('digit_recognition_cnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


# Load the model once
model = load_model()


def predict_digit(image):
    """
    Predict the digit from the drawn image
    
    Args:
        image: PIL Image or numpy array from the Gradio canvas
    
    Returns:
        Dictionary with digit predictions and confidence scores
    """
    if image is None:
        return "Please draw a digit first!"
    
    try:
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle different image formats
        if len(image.shape) == 3:
            # If RGB, convert to grayscale
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, 3]  # Use alpha channel
            else:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Invert if background is white (Gradio default)
        if image.mean() > 127:
            image = 255 - image
        
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        # Create confidence dictionary for all digits
        confidences = {str(i): float(probabilities[0][i].item()) for i in range(10)}
        
        return confidences
        
    except Exception as e:
        return f"Error processing image: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Hand Digit Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ✍️ Hand Digit Recognition
        ### Draw a digit (0-9) and let the AI recognize it!
        
        This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset.
        """
    )
    
    with gr.Row():
        with gr.Column():
            # Drawing canvas
            canvas = gr.Sketchpad(
                label="Draw a digit here",
                type="numpy",
                image_mode="L",
                canvas_size=(280, 280),
                brush=gr.Brush(default_size=15, colors=["#FFFFFF"], color_mode="fixed")
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                predict_btn = gr.Button("🔮 Predict", variant="primary")
        
        with gr.Column():
            # Output
            output = gr.Label(label="Prediction Confidence", num_top_classes=10)
            
            gr.Markdown(
                """
                ### 📊 Model Information
                - **Model Type:** Convolutional Neural Network (CNN)
                - **Training Dataset:** MNIST (60,000 images)
                - **Input Size:** 28x28 pixels
                - **Classes:** 10 digits (0-9)
                - **Accuracy:** ~95%+
                """
            )
    
    gr.Markdown(
        """
        ### 💡 Tips for Best Results:
        - Draw the digit clearly in the center of the canvas
        - Use a single continuous stroke when possible
        - Make the digit fill most of the canvas
        - Try different drawing styles if the prediction isn't accurate
        """
    )
    
    # Button actions
    predict_btn.click(fn=predict_digit, inputs=canvas, outputs=output)
    clear_btn.click(fn=lambda: None, inputs=None, outputs=canvas)
    
    # Also predict on canvas change for real-time feedback
    canvas.change(fn=predict_digit, inputs=canvas, outputs=output)


# Launch the app
if __name__ == "__main__":
    demo.launch()
