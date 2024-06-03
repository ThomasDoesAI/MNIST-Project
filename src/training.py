import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loading import load_mnist_data
from models import MNISTModel


def train_model(model, train_loader, epochs=20, learning_rate=0.001):
    optomizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optomizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optomizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == "__main__":
    train_loader, _ = load_mnist_data()
    model = MNISTModel()
    train_model(model, train_loader)
    torch.save(model.state_dict(), 'models/mnist_model.pth')