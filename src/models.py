import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        #self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.batch_norm1(F.relu(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.batch_norm2(F.relu(self.fc2(x)))
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x
