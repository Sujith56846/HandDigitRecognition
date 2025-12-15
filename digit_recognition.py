import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

import torch
import torch.nn as nn
import pygame
import numpy as np
import cv2
import os


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


def load_model():
    """Load the pre-trained model (CNN preferred, ANN as fallback)"""
    cnn_model_path = 'digit_recognition_cnn_model.pth'
    ann_model_path = 'digit_recognition_ann_model.pth'
    
    # Try to load CNN model first (best performance)
    if os.path.exists(cnn_model_path):
        model = CNN()
        model.load_state_dict(torch.load(cnn_model_path))
        model.eval()
        print(f"CNN Model loaded successfully from {cnn_model_path}")
        return model, "CNN"
    
    # Fallback to ANN model
    elif os.path.exists(ann_model_path):
        model = ANN()
        model.load_state_dict(torch.load(ann_model_path))
        model.eval()
        print(f"ANN Model loaded successfully from {ann_model_path}")
        return model, "ANN"
    
    # No model found
    else:
        print("No trained model found!")
        print("Available model files to look for:")
        print(f"  - {cnn_model_path} (CNN - Best performance)")
        print(f"  - {ann_model_path} (ANN - Good performance)")
        print("\nPlease run 'train_model.py' first to train and save the models.")
        return None, None


def draw_digit(model, model_type):
    """Pygame interface for drawing digits and getting predictions"""
    if model is None:
        print("Cannot start drawing interface without a trained model.")
        return
    
    pygame.init()
    window_size = 280  # 10X scale of 28x28
    display_height = window_size + 100  # More space for UI elements
    screen = pygame.display.set_mode((window_size, display_height))
    pygame.display.set_caption(f"Hand Digit Recognition ({model_type}) - Draw 0-9")
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    drawing = False
    prediction = None
    confidence = 0.0

    # Fonts for different text elements
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)

    print("Drawing interface started!")
    print("Instructions:")
    print("- Draw a digit with your mouse")
    print("- Press 'C' to clear the screen")
    print("- Close the window to exit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                # call function for prediction
                prediction, confidence = predict_digit(model, screen)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0, 0, 0))
                    prediction = None
                    confidence = 0.0

            if event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.circle(screen, (255, 255, 255), event.pos, 8)

        # Clear the bottom area for UI elements
        pygame.draw.rect(screen, (0, 0, 0), (0, window_size, window_size, 100))
        
        # Display prediction and confidence
        if prediction is not None:
            # Main prediction
            pred_text = font_large.render(f"Digit: {prediction}", True, (0, 255, 0))
            screen.blit(pred_text, (10, window_size + 10))
            
            # Confidence score
            conf_text = font_medium.render(f"Confidence: {confidence:.1f}%", True, (255, 255, 0))
            screen.blit(conf_text, (10, window_size + 50))
        else:
            # Instructions when no prediction
            instr_text = font_medium.render("Draw a digit (0-9)", True, (200, 200, 200))
            screen.blit(instr_text, (10, window_size + 10))
        
        # Model info and instructions
        model_text = font_small.render(f"Model: {model_type}", True, (150, 150, 150))
        clear_text = font_small.render("Press 'C' to clear", True, (150, 150, 150))
        screen.blit(model_text, (10, window_size + 75))
        screen.blit(clear_text, (150, window_size + 75))

        pygame.display.flip()
        clock.tick(60)


def process_drawing(screen):
    """Process the pygame screen drawing into a format suitable for the model"""
    surface = pygame.surfarray.array3d(screen)
    gray = np.dot(surface[..., :3], [0.2989, 0.587, 0.114])  # converts to grayscale
    gray = np.transpose(gray, (1, 0))  # Transpose to match MNIST orientation
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    gray = (gray - 0.5) / 0.5  # Normalize to [-1, 1]
    tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    return tensor


def predict_digit(model, screen):
    """Predict the digit drawn on the screen with confidence score"""
    image = process_drawing(screen)
    if image is None:
        return None, 0.0

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    return prediction.item(), confidence.item() * 100


if __name__ == "__main__":
    # Load the pre-trained model
    model, model_type = load_model()
    
    # Start the drawing interface
    if model is not None:
        print(f"\nUsing {model_type} model for digit recognition")
        print("Note: CNN models typically provide better accuracy than ANN models")
        draw_digit(model, model_type)
    else:
        print("\nCannot start the application without a trained model.")