import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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


file      = 'dataset/Data/control_2024-12-04-15-29-53.txt'
pose_file = 'dataset/Data/pose_2024-12-04-15-29-53.txt'
imu_file  = 'dataset/Data/imu_2024-12-04-15-29-53.txt'
# file = 'Test_Data/rsm_2023-10-19-11-31-36.txt'

actuator_t_delay = 0.05
data = np.loadtxt(file)
time = data[:,0] 
time_begin = time[0]
time  = (time - time_begin)/1e9 + actuator_t_delay

bodyrate_x = data[:,1]
bodyrate_y = data[:,2]
bodyrate_z = data[:,3]
thrust     = data[:,4]

# Thrust + Bodyrate
thrust     = signal.filtfilt(b3, a3, thrust)
bodyrate_x = signal.filtfilt(b3, a3, bodyrate_x)
bodyrate_y = signal.filtfilt(b3, a3, bodyrate_y)
bodyrate_z = signal.filtfilt(b3, a3, bodyrate_z)

pose_data = np.loadtxt(pose_file)
pose_time = pose_data[:,0] 
pose_time = (pose_time - time_begin)/1e9
pose_x    = pose_data[:,1]
pose_y    = pose_data[:,2]
pose_z    = pose_data[:,3]

q_x    = pose_data[:,4]
q_y    = pose_data[:,5]
q_z    = pose_data[:,6]
q_w    = pose_data[:,7]

time_duration = (time[1:-1] - time[0:-2])/1e6

imu_data = np.loadtxt(imu_file)
imu_time = imu_data[:,0] 
imu_time = (imu_time - time_begin) / 1e9

# 1: static or moving box (plane) collision; 2: stick collision; 3: Landing collision
# collision_type = [1,    1,    1,     2,     2,     2,     2,     2,     2,     2,     2,     2,    1,       1,      1,      1,     1,     1,       3]
# collision_time = [23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 67.73, 71.73, 75.66, 80.67, 85.6, 102.091, 111.87, 126.54, 142.09, 155.7, 168.024, 181.15]


collision_type = [3,   3,   3,   3,    3,    3,  3,   3,   3,   3,   3,   3,    1,    1,    1,     2,     2,     2,     2,     2,     2,     2,     2,     2,    1,      1,      1,      1,      1,     1,      ]
collision_time = [2.5, 5.0, 6.5, 10.0, 17.5, 93, 108, 149, 161, 119, 134, 91.8, 23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 67.73, 71.73, 75.66, 80.67, 85.6, 103.17, 111.87, 126.54, 142.09, 155.7, 168.024]



# collision_type = [1,    1,    1,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,    1,       1,      1,      1,     1,     1,       3]
# collision_time = [23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 67.73, 71.73, 75.66, 80.67, 81.21, 82.19, 85.6, 102.091, 111.87, 126.54, 142.2, 155.7, 168.024, 181.151]

sample_index = 5
duration = 0.5
offset   = 0.05
start_time = collision_time[sample_index] - duration + offset
end_time   = collision_time[sample_index] + offset

start_index = find_index(imu_time, start_time)
end_index   = find_index(imu_time, end_time)

angular_velocity_x = imu_data[:,1]
angular_velocity_y = imu_data[:,2]
angular_velocity_z = imu_data[:,3]

linear_acceleration_x = imu_data[:,4]
linear_acceleration_y = imu_data[:,5]
linear_acceleration_z = imu_data[:,6]

# Angular velocity
angular_velocity_x = signal.filtfilt(b2, a2, angular_velocity_x)
angular_velocity_y = signal.filtfilt(b2, a2, angular_velocity_y)
angular_velocity_z = signal.filtfilt(b2, a2, angular_velocity_z)

# Linear Acceleration
linear_acceleration_x = signal.filtfilt(b2, a2, linear_acceleration_x)
linear_acceleration_y = signal.filtfilt(b2, a2, linear_acceleration_y)
linear_acceleration_z = signal.filtfilt(b2, a2, linear_acceleration_z)



# ... (previous data preparation code remains unchanged until here)

# Custom Dataset
class CollisionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Prepare data for all collisions
X_data = []
y_data = []
for i in range(len(collision_time)):
    start_time = collision_time[i] - duration + offset
    end_time = collision_time[i] + offset
    start_index = find_index(imu_time, start_time)
    end_index = find_index(imu_time, end_time)
    
    if start_index != -1 and end_index != -1:
        X = np.stack([
            linear_acceleration_x[start_index:end_index],
            linear_acceleration_y[start_index:end_index],
            linear_acceleration_z[start_index:end_index],
            angular_velocity_x[start_index:end_index],
            angular_velocity_y[start_index:end_index],
            angular_velocity_z[start_index:end_index]
        ], axis=1)  # Shape: (time_steps, features)
        
        # Ensure fixed number of time steps
        if X.shape[0] > int(fs_imu * duration):
            X = X[:int(fs_imu * duration)]
        elif X.shape[0] < int(fs_imu * duration):
            X = np.pad(X, ((0, int(fs_imu * duration) - X.shape[0]), (0, 0)), mode='constant')
        
        X_data.append(X)
        y_data.append(collision_type[i] - 1)  # Adjust labels to start from 0 for classification

# Convert lists to numpy arrays, then to PyTorch tensors
X_data = np.array(X_data)
y_data = np.array(y_data)

# Split data into train and validation sets (example split here is 80% train, 20% validation)
train_size = int(0.8 * len(X_data))
train_dataset = CollisionDataset(X_data[:train_size], y_data[:train_size])
val_dataset = CollisionDataset(X_data[train_size:], y_data[train_size:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the CNN model
class CollisionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CollisionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (int(fs_imu * duration) // 4), 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = len(set(collision_type))
model = CollisionClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 2, 1))  # Permute to match expected input shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.permute(0, 2, 1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * correct / total:.2f}%')

# Predict for sample data (using the last example from the loop)
sample_data = torch.FloatTensor(X_data[-7]).unsqueeze(0)  # Add batch dimension
model.eval()
with torch.no_grad():
    outputs = model(sample_data.permute(0, 2, 1))
    _, predicted = torch.max(outputs.data, 1)
    print(f'Predicted collision type: {predicted.item() + 1}')

####################################################################################################
########################################### Plot Figures ###########################################
####################################################################################################


colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bottom_colors = ['#d62728', '#9467bd', '#8c564b']  # Red, purple, brown

plt.figure(figsize=(3.7, 2.7))
plt.plot(imu_time[start_index:end_index], linear_acceleration_y[start_index:end_index], color=colors[1], linestyle='-', linewidth=1)
plt.plot(imu_time[start_index:end_index], linear_acceleration_z[start_index:end_index], color=colors[2], linestyle='-', linewidth=1)
plt.plot(imu_time[start_index:end_index], linear_acceleration_x[start_index:end_index], color=colors[0], linestyle='-', linewidth=1)
plt.legend(['Y', 'Z', 'X'], frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('Linear acceleration (m/s2)')
plt.title('Linear acceleration sample')   

plt.figure(figsize=(3.7, 2.7))
# plt.grid(True, linewidth = 2, color='gray', alpha=0.8)
plt.plot(imu_time[start_index:end_index], angular_velocity_y[start_index:end_index], color=colors[1], linestyle='-', linewidth=1)
plt.plot(imu_time[start_index:end_index], angular_velocity_z[start_index:end_index], color=colors[2], linestyle='-', linewidth=1)
plt.plot(imu_time[start_index:end_index], angular_velocity_x[start_index:end_index], color=colors[0], linestyle='-', linewidth=1)
# plt.xlim(left, right)
# plt.ylim(-2.5, 2.5)
plt.legend(['Y', 'Z', 'X'], frameon=False)
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Angular velocity sample')   

plt.show()

