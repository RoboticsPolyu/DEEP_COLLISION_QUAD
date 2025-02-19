import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
from scipy import signal

# Constants
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bottom_colors = ['#d62728', '#9467bd', '#8c564b']  # Red, purple, brown

fs         = 300  # Sampling frequency
fs_imu     = 150
fs_control = 100
cutoff     = 30  # Cutoff frequency in Hz
order      = 5  # Order of the filter

# Normalize the frequency
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

# Get the filter coefficients
b, a = signal.butter(order, normal_cutoff, btype='low', analog=False) # for pose data
b2, a2 = signal.butter(order, cutoff / (0.5 * fs_imu), btype='low', analog=False) # for imu data
b3, a3 = signal.butter(order, cutoff / (0.5 * fs_control), btype='low', analog=False) # for control data

def find_index(time_sequence, time0):
    for index, time in enumerate(time_sequence):
        if time >= time0:
            return index
    return -1  # Return -1 if no such time is found

# Load data (assuming these files exist and contain the correct data format)
file      = 'dataset/Data/control_2024-12-04-15-29-53.txt'
pose_file = 'dataset/Data/pose_2024-12-04-15-29-53.txt'
imu_file  = 'dataset/Data/imu_2024-12-04-15-29-53.txt'

actuator_t_delay = 0.05
data = np.loadtxt(file)
time = data[:,0] 
time_begin = time[0]
time  = (time - time_begin)/1e9 + actuator_t_delay

# ... (Data processing for control, pose, and imu as in the original script)

imu_data = np.loadtxt(imu_file)
imu_time = imu_data[:,0] 
imu_time = (imu_time - time_begin) / 1e9

collision_type = [3,  3,   3,   1,    1,    1,     2,     2,     2,     2,     3,   3,   3,   3,    3,    2,     2,     2,     2,     2,    1,      1,      1,      1,      1,     1,       3,   3,   3]
collision_time = [93, 149, 161, 23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 2.5, 5.0, 6.5, 10.0, 17.5, 67.73, 71.73, 75.66, 80.67, 85.6, 103.17, 111.87, 126.54, 142.09, 155.7, 168.024, 119, 134, 91.8]

# Prepare data for all collisions
duration = 0.10
offset   = 0.05
X_data = []
y_data = []
for i in range(len(collision_time)):
    start_time = collision_time[i] - duration + offset
    end_time = collision_time[i] + offset
    start_index = find_index(imu_time, start_time)
    end_index = find_index(imu_time, end_time)
    
    if start_index != -1 and end_index != -1:
        sample = np.stack([
            imu_data[start_index:end_index, 4],  # linear_acceleration_x
            imu_data[start_index:end_index, 5],  # linear_acceleration_y
            imu_data[start_index:end_index, 6],  # linear_acceleration_z
            imu_data[start_index:end_index, 1],  # angular_velocity_x
            imu_data[start_index:end_index, 2],  # angular_velocity_y
            imu_data[start_index:end_index, 3]   # angular_velocity_z
        ], axis=1)
        
        # Ensure fixed number of features
        fixed_time_steps = int(fs_imu * duration)
        if sample.shape[0] > fixed_time_steps:
            sample = sample[:fixed_time_steps]
        elif sample.shape[0] < fixed_time_steps:
            padding = np.zeros((fixed_time_steps - sample.shape[0], 6))
            sample = np.vstack([sample, padding])
        
        X_data.append(sample)
        y_data.append(collision_type[i] - 1)  # Adjust labels to start from 0 for classification

# Convert lists to numpy arrays, then to PyTorch tensors
X_data = np.array(X_data)
y_data = np.array(y_data)

# Custom Dataset for MobileNet
class CollisionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split data into train and validation sets
train_size = int(0.8 * len(X_data))
train_dataset = CollisionDataset(X_data[:train_size], y_data[:train_size])
val_dataset = CollisionDataset(X_data[train_size:], y_data[train_size:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the MobileNet model
class CollisionClassifierMobileNet(nn.Module):
    def __init__(self, num_classes):
        super(CollisionClassifierMobileNet, self).__init__()
        self.mobilenet = mobilenet_v2(weights='DEFAULT')  # False because we're dealing with new data type
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.mobilenet.children())[:-1])
        # Adjust for our input (1 channel for time series)
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Add new classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)  # 1280 is the output size from MobileNetV2 before the final classification layer
        )
        
    def forward(self, x):
        # Reshape input: (batch_size, time_steps, features) -> (batch_size, 1, features, time_steps)
        x = x.view(x.size(0), 1, 6, -1)  # Assuming 6 features per time step
        x = self.features(x)
        # Global average pooling
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = len(set(collision_type))
model = CollisionClassifierMobileNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * correct / total:.2f}%')

# Plotting misclassifications (simplified for brevity)
model.eval()
misclassified_predictions = []
with torch.no_grad():
    inputs, labels = val_dataset[:]
    outputs = model(torch.FloatTensor(inputs))
    _, predicted = torch.max(outputs.data, 1)
    
    for i, (pred, true) in enumerate(zip(predicted, labels)):
        if pred != true:
            misclassified_predictions.append((i, true.item() + 1, pred.item() + 1))  # Add 1 to get back to original class labels

print("Misclassified samples:")
for index, true_label, predicted_label in misclassified_predictions:
    print(f"Sample index {index}: True label {true_label}, Predicted label {predicted_label}")

plt.show()