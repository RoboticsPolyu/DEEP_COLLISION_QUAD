import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

# Step 1: Define the Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Step 2: Define the DCD model
class DCD(nn.Module):
    def __init__(self):
        super(DCD, self).__init__()
        self.tconv1 = nn.Conv1d(7, 64, kernel_size=3, padding=1)
        self.tconv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.tconv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.tconv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.tconv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.tconv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.tconv7 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 10, 3)  # Output 3-dimensional vector

    def forward(self, x):
        x = x.transpose(1, 2)  # Reshape to [batch_size, num_channels, length]
        x = torch.relu(self.tconv1(x))
        x = torch.relu(self.tconv2(x))
        x = torch.relu(self.tconv3(x))
        x = torch.relu(self.tconv4(x))
        x = torch.relu(self.tconv5(x))
        x = torch.relu(self.tconv6(x))
        x = torch.relu(self.tconv7(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# Step 3: Training and Testing Functions
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# Parameters
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# Create dataset and dataloaders

data   = np.random.randn(1000, 10, 7)   # Example data
labels = np.random.randint(0, 3, 1000)  # Example labels

print('-------------------------')
print(torch.Tensor(data)[50])

print('-------------------------')
print(torch.LongTensor(labels))

dataset = TimeSeriesDataset(torch.Tensor(data), torch.LongTensor(labels))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")

model = DCD().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and testing loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), 'model_parameters.pth')

# plt.figure(figsize=(3.7, 3.0))
# # Plot the training loss
# plt.plot(range(1, num_epochs+1), train_loss, 'b-', label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.legend()