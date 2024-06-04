import torch
from data_loading import load_mnist_data  # Ensure you have this module to load MNIST data
from models import MNISTModel  # Import the updated MNISTModel from your models file
import torch.nn.functional as F

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0  # Initialize test loss
    correct = 0  # Initialize the count of correct predictions

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in test_loader:
            output = model(data)  # Forward pass: compute the output of the model on the input data
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Calculate average test loss
    accuracy = correct / len(test_loader.dataset)  # Calculate accuracy
    print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')  # Print the test loss and accuracy

if __name__ == "__main__":
    _, test_loader = load_mnist_data()  # Load the test data using a custom data loading function
    model = MNISTModel()  # Initialize the model
    model.load_state_dict(torch.load('models/mnist_model.pth'))  # Load the trained model's state dictionary
    evaluate_model(model, test_loader)  # Evaluate the model with the test data loader
