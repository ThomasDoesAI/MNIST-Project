import torch
from data_loading import load_mnist_data
from models import MNISTModel
import torch.nn.functional as F

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    _, test_loader = load_mnist_data()
    model = MNISTModel()
    model.load_state_dict(torch.load('models/mnist_model.pth'))
    evaluate_model(model, test_loader)