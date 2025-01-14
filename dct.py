import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.fc = nn.Linear(128, 3)  # Assuming three classes
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        x = self.fc(x)
        
        return x

# Example Training Code
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Example usage
if __name__ == "__main__":
    # Generate some example data
    batch_size = 100
    time_steps = 10
    
    device = torch.device("mps")
    # Generating sample input with 10 thrusts, 10 acceleration (3D), 10 gyroscope (3D)
    inputs = torch.randn(batch_size, 20, time_steps)
    labels = torch.randint(0, 3, (batch_size,))  # 100 labels with three classes (0, 1, 2)
    
    # Create a DataLoader
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize the model, criterion, and optimizer
    model = TCN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=20)

