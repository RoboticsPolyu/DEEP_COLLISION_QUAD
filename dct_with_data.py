import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Parameters
num_samples  = 100  # Number of samples
num_features = 7    # Number of features per sample
num_classes  = 4    # Number of classes
batch_num    = 10

# Font
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
plt.rcParams['axes.labelsize'] = 8  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Define a color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bottom_colors = ['#d62728', '#9467bd', '#8c564b']  # Red, purple, brown

# Function to set dense grid
def set_dense_grid(ax):
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', alpha=0.5)
    ax.minorticks_on()  # Enable minor ticks

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(num_features, 64, kernel_size=3, dilation=1, padding=1),
            ResidualBlock(64,  64,  kernel_size=3, dilation=2, padding=2),
            ResidualBlock(64,  64,  kernel_size=3, dilation=4, padding=4),
            ResidualBlock(64,  64,  kernel_size=3, dilation=8, padding=8),
            ResidualBlock(64,  128, kernel_size=3, dilation=1, padding=1),
            ResidualBlock(128, 128, kernel_size=3, dilation=2, padding=2),
            ResidualBlock(128, 128, kernel_size=3, dilation=4, padding=4)
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.residual_blocks(x)
        out = torch.mean(out, dim=2)  # Global average pooling
        out = self.fc(out)
        return out
    
# Training Function
def train(model, train_loader, criterion, optimizer, num_epochs):
    epoch_losses = []
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
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return epoch_losses

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Print classification report
    print(classification_report(all_labels, all_preds))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(3.7, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main code for training and evaluating
if __name__ == "__main__":
    # Load and preprocess your data
    data = np.loadtxt('test_data.txt', delimiter=',')
    labels = np.loadtxt('label.txt', delimiter=',')

    batch_size = data.shape[0] // batch_num
    inputs = data.reshape(batch_size, num_features, batch_num)
    print('---------------------------------------')
    print(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    print('---------------------------------------')
    print(inputs)

    labels = torch.tensor(labels[:batch_size], dtype=torch.long) # Make sure labels match the new batch size
    print('---------------------------------------')
    print(labels)
    print("Inputs shape:", inputs.shape)
    print("Labels shape:", labels.shape)

    dataset = TensorDataset(inputs, labels)

    # Split dataset into training and testing sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset,   batch_size=32, shuffle=False)

    # Initialize the model, criterion, and optimizer
    model = TCN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 20
    epoch_losses = train(model, train_loader, criterion, optimizer, num_epochs)
    
    torch.save(model.state_dict(), 'model/model_parameters.pth')
    
    plt.figure(figsize=(3.7, 3.0))
    set_dense_grid(plt)
    # Plot the training loss
    plt.plot(range(1, num_epochs+1), epoch_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    
    # Evaluate the model
    evaluate_model(model, test_loader)
