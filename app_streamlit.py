import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
   
# Your model classes here...
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
        # After 3 pooling operations: 28x28 -> 14x14 -> 7x7 -> 3x3 (with padding)
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

def main():
    st.title("Hand Digit Recognition")
       
    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
       )
       
    if canvas_result.image_data is not None:
        # Process and predict
        prediction = process_canvas_image(canvas_result.image_data)
        st.write(f"Predicted Digit: {prediction}")
   
if __name__ == "__main__":
    main()