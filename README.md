# Hand Digit Recognition Project

A real-time hand digit recognition system using PyTorch neural networks (CNN and ANN) with an interactive pygame canvas interface.

## 🚀 Features

- **Interactive Canvas**: Draw digits with your mouse on a 280x280 pixel canvas
- **Real-time Prediction**: Get instant predictions as you finish drawing
- **Multiple Model Support**: 
  - CNN Model (Convolutional Neural Network) - Higher accuracy ~95-99%
  - ANN Model (Artificial Neural Network) - Good accuracy ~90-95%
- **Smart Model Loading**: Automatically selects the best available trained model
- **User-friendly Interface**: Simple controls with visual feedback

## 📋 Requirements

- Python 3.8+
- PyTorch
- torchvision
- pygame
- numpy
- opencv-python (cv2)

## 🛠️ Installation

1. Clone or download this project
2. Install required packages:
   ```bash
   pip install torch torchvision pygame numpy opencv-python
   ```

## 🎯 Quick Start

### Step 1: Train the Models
First, train both CNN and ANN models:
```bash
python train_model.py
```

This will:
- Download the MNIST dataset (~50MB)
- Train both CNN and ANN models (10 epochs each)
- Save models as `digit_recognition_cnn_model.pth` and `digit_recognition_ann_model.pth`
- Display training progress and final accuracy

### Step 2: Run the Digit Recognition Interface
```bash
python digit_recognition.py
```

## 🎮 How to Use

1. **Start the Application**: Run `python digit_recognition.py`
2. **Draw a Digit**: 
   - Use your mouse to draw a digit (0-9) on the black canvas
   - The drawing area is 280x280 pixels (scaled 10x from MNIST 28x28)
3. **Get Prediction**: 
   - Release the mouse button to get an instant prediction
   - The predicted digit appears in green text below the canvas
4. **Clear Canvas**: Press 'C' key to clear the canvas and draw a new digit
5. **Exit**: Close the window to exit the application

## 🏗️ Project Structure

```
BangaloreHousePrediction/
│
├── train_model.py              # Model training script
├── digit_recognition.py        # Main application with GUI
├── requirements.txt           # Python dependencies
│
├── data/                      # MNIST dataset (auto-downloaded)
│   └── MNIST/
│       └── raw/
│
├── digit_recognition_cnn_model.pth  # Trained CNN model
├── digit_recognition_ann_model.pth  # Trained ANN model
│
└── README.md                  # This file
```

## 🧠 Model Architecture

### CNN Model (Recommended)
- 3 Convolutional layers (32→64→128 filters)
- Max pooling layers for dimension reduction
- Dropout layers (25%) for regularization
- Fully connected layers: 512→128→10
- **Expected Accuracy**: 95-99%

### ANN Model (Fallback)
- Fully connected neural network
- 4 layers: 784→128→128→128→10
- ReLU activation functions
- **Expected Accuracy**: 90-95%

## 🔧 Customization

### Adjust Model Parameters
Edit `train_model.py` to modify:
- Number of training epochs (default: 10)
- Learning rate (default: 0.001)
- Batch size (default: 64)
- Network architecture

### Modify Interface
Edit `digit_recognition.py` to change:
- Canvas size (default: 280x280)
- Brush size (default: 8 pixels)
- Colors and fonts
- Window layout

## 🚀 Deployment Options

### 1. Standalone Executable
Create a standalone executable using PyInstaller:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed digit_recognition.py
```

### 2. Web Application
Convert to a web app using Streamlit or Flask:
- Replace pygame with web canvas (HTML5 Canvas)
- Use base64 image encoding for model input
- Deploy on Heroku, AWS, or similar platforms

### 3. Mobile App
Use frameworks like:
- **Kivy**: Cross-platform Python mobile apps
- **React Native**: With Python backend API
- **Flutter**: With TensorFlow Lite model conversion

## 🎯 Performance Tips

1. **Use CNN Model**: Always prefer CNN over ANN for better accuracy
2. **Drawing Quality**: Draw digits clearly and large enough to fill most of the canvas
3. **Model Training**: Train for more epochs for better accuracy (trade-off with time)
4. **Data Augmentation**: Add rotation, scaling, and noise to training data

## 🐛 Troubleshooting

### Model Not Found Error
- Make sure you've run `train_model.py` first
- Check if `.pth` files exist in the project directory

### Poor Recognition Accuracy
- Draw digits more clearly and boldly
- Try drawing larger digits that fill more of the canvas
- Ensure digits are centered and similar to MNIST style

### Pygame Import Error
```bash
pip install pygame
```

### PyTorch Import Error
```bash
pip install torch torchvision
```

## 📊 Expected Results

- **Training Time**: 2-5 minutes on modern CPU
- **Model Size**: ~1-5MB each
- **Inference Speed**: Real-time (<50ms per prediction)
- **Accuracy**: 95-99% on well-drawn digits

## 🤝 Contributing

1. Fork the project
2. Create your feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

**Enjoy drawing and recognizing digits! 🎨🔢**