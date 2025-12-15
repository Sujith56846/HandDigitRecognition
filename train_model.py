import warnings
# Suppress the pkg_resources deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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


def train_model():
    print("Starting MNIST digit recognition model training...")
    
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load MNIST dataset (suppress pkg_resources warnings)
    print("Loading MNIST dataset...")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print("Dataset loaded successfully!")

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Initialize both models
    ann_model = ANN()
    cnn_model = CNN()
    
    loss_fn = nn.CrossEntropyLoss()
    ann_optimizer = optim.Adam(ann_model.parameters(), lr=0.001)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    episodes = 10
    
    # Train ANN Model
    print(f"\n=== Training ANN Model for {episodes} epochs ===")
    for epoch in range(episodes):
        ann_model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            ann_optimizer.zero_grad()
            outputs = ann_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            ann_optimizer.step()
            running_loss += loss.item()
        print(f"ANN Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    # Test ANN Model
    print("\nTesting ANN model accuracy...")
    ann_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = ann_model(images)
            _, prediction = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    
    ann_accuracy = 100 * correct / total
    print(f"ANN Final Accuracy: {ann_accuracy:.2f}%")

    # Train CNN Model
    print(f"\n=== Training CNN Model for {episodes} epochs ===")
    for epoch in range(episodes):
        cnn_model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            cnn_optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            cnn_optimizer.step()
            running_loss += loss.item()
        print(f"CNN Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    # Test CNN Model
    print("\nTesting CNN model accuracy...")
    cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = cnn_model(images)
            _, prediction = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    
    cnn_accuracy = 100 * correct / total
    print(f"CNN Final Accuracy: {cnn_accuracy:.2f}%")

    # Save both trained models
    ann_model_path = 'digit_recognition_ann_model.pth'
    cnn_model_path = 'digit_recognition_cnn_model.pth'
    
    torch.save(ann_model.state_dict(), ann_model_path)
    torch.save(cnn_model.state_dict(), cnn_model_path)
    
    print(f"\nModels saved:")
    print(f"ANN Model: {ann_model_path} (Accuracy: {ann_accuracy:.2f}%)")
    print(f"CNN Model: {cnn_model_path} (Accuracy: {cnn_accuracy:.2f}%)")
    print("Training completed successfully!")

    return ann_model, cnn_model, ann_accuracy, cnn_accuracy


if __name__ == "__main__":
    train_model()