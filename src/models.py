import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel, padding of 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces the size by a factor of 2
        )
        
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel, padding of 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Further reduces the size by a factor of 2
        )

        # Fully connected layer: the output size needs to match the input from the flattened feature map
        # After two 2x2 poolings on a 28x28 image, the size is reduced to 7x7
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Input x shape: [batch_size, 1, 28, 28]
        x = self.conv1(x)  # Apply the first convolutional layer
        x = self.conv2(x)  # Apply the second convolutional layer
        x = x.view(x.size(0), -1)  # Flatten the tensor to [batch_size, 64*7*7]
        x = self.fc1(x)  # Apply the first fully connected layer
        x = self.fc2(x)  # Apply the second fully connected layer
        x = self.fc3(x)  # Apply the output layer
        return F.log_softmax(x, dim=1)  # Apply log_softmax for the output
