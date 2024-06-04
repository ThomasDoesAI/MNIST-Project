import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loading import load_mnist_data  # Ensure you have this module to load MNIST data
from models import MNISTModel  # Import the updated MNISTModel from your models file

def train_model(model, train_loader, epochs=20, learning_rate=0.001, weight_decay=1e-4):
    # Initialize the Adam optimizer with the model's parameters, learning rate, and weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Initialize a learning rate scheduler to reduce the learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Loop through the specified number of epochs
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear the gradients of all optimized tensors
            output = model(data)  # Forward pass: compute the output of the model on the input data
            loss = F.cross_entropy(output, target)  # Compute the loss using cross-entropy
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step (parameter update)
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # Print the loss for the current epoch
        scheduler.step(loss.item())  # Step the learning rate scheduler with the current loss

if __name__ == "__main__":
    # Load the training data using a custom data loading function
    train_loader, _ = load_mnist_data()
    # Initialize the model
    model = MNISTModel()
    # Train the model with the training data loader
    train_model(model, train_loader)
    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), 'models/mnist_model.pth')
