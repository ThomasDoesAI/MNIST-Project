import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loading import load_mnist_data
from models import MNISTModel


def train_model(model, train_loader, epochs=30, learning_rate=0.001, weight_decay=1e-4):
    optomizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optomizer, mode='min', factor=0.1, patience=3)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optomizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optomizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        scheduler.step(loss.item())

if __name__ == "__main__":
    train_loader, _ = load_mnist_data()
    model = MNISTModel()
    train_model(model, train_loader)
    torch.save(model.state_dict(), 'models/mnist_model.pth')