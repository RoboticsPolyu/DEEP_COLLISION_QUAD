import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

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

# File paths (assuming these exist in your environment)
file      = 'dataset/Data/control_2024-12-04-15-29-53.txt'
pose_file = 'dataset/Data/pose_2024-12-04-15-29-53.txt'
imu_file  = 'dataset/Data/imu_2024-12-04-15-29-53.txt'

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

collision_type = [3,  3,   3,   1,    1,    1,     2,     2,     2,     2,     3,   3,   3,   3,    3,    2,     2,     2,     2,     2,    1,      1,      1,      1,      1,     1,       3,   3,   3]
collision_time = [93, 149, 161, 23.5, 29.5, 36.43, 49.44, 52.68, 60.30, 65.23, 2.5, 5.0, 6.5, 10.0, 17.5, 67.73, 71.73, 75.66, 80.67, 85.6, 103.17, 111.87, 126.54, 142.09, 155.7, 168.024, 119, 134, 91.8]

sample_index = 5
duration = 2.0
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

# Custom Dataset for MLP
class CollisionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Prepare data for all collisions with interpolation
X_data = []
y_data = []
for i in range(len(collision_time)):
    start_time = collision_time[i] - duration + offset
    end_time   = collision_time[i] + offset
    start_index = find_index(imu_time, start_time)
    end_index   = find_index(imu_time, end_time)
    
    if start_index != -1 and end_index != -1:
        # Time points for interpolation
        original_time = imu_time[start_index:end_index]
        new_time = np.linspace(start_time, end_time, int(fs_imu * duration))
        
        # Data to interpolate, now including thrust
        data_to_interpolate = [
            linear_acceleration_x[start_index:end_index],
            linear_acceleration_y[start_index:end_index],
            linear_acceleration_z[start_index:end_index],
            angular_velocity_x[start_index:end_index],
            angular_velocity_y[start_index:end_index],
            angular_velocity_z[start_index:end_index],
            thrust[find_index(time, start_time):find_index(time, end_time)]  # Assuming 'time' corresponds to 'thrust'
        ]
        
        # Interpolate each data series
        interpolated_data = []
        for idx, data_series in enumerate(data_to_interpolate):
            if idx < 6:  # For IMU data, we have time points from 'imu_time'
                f = interp1d(original_time, data_series, kind='linear', fill_value="extrapolate")
                interpolated_data.append(f(new_time))
            else:  # For thrust, we'll use 'time' for interpolation
                thrust_time = time[find_index(time, start_time):find_index(time, end_time)]
                f = interp1d(thrust_time, data_series, kind='linear', fill_value="extrapolate")
                interpolated_data.append(f(new_time))

        # Stack and flatten the interpolated data
        X = np.column_stack(interpolated_data).flatten()
        
        # Update fixed_features to account for thrust
        fixed_features = int(fs_imu * duration) * 7  # Now 7 features including thrust
        if len(X) != fixed_features:
            raise ValueError(f"Expected {fixed_features} features but got {len(X)}")
        
        X_data.append(X)
        y_data.append(collision_type[i] - 1)  # Adjust labels to start from 0 for classification

# Convert lists to numpy arrays, then to PyTorch tensors
X_data = np.array(X_data)
y_data = np.array(y_data)

# Create full dataset
full_dataset  = CollisionDataset(X_data, y_data)

# Randomly split the dataset into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=32)

# Define the MLP model with updated input size
class CollisionClassifierMLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CollisionClassifierMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = len(set(collision_type))
input_size = fixed_features  # This should match your flattened input size
model = CollisionClassifierMLP(num_classes, input_size)
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
    misclassified = []  # Store indices of misclassified samples

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store indices of misclassified samples
            misclassified.extend([j for j, (pred, true) in enumerate(zip(predicted, labels)) 
                                  if pred != true])

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * correct / total:.2f}%')

# After training, evaluate the model on the validation set again to get detailed misclassifications
model.eval()
misclassified_predictions = []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for j, (pred, true) in enumerate(zip(predicted, labels)):
            # if pred != true:
                misclassified_predictions.append((i * 32 + j, true.item() + 1, pred.item() + 1))  # Add 1 to get back to original class labels

print("Misclassified samples:")
for index, true_label, predicted_label in misclassified_predictions:
    print(f"Sample index {index}: True label {true_label}, Predicted label {predicted_label}")

    # Plot the misclassified sample
    actual_index = val_dataset.indices[index]  # Get the actual index in the full dataset

    sample = X_data[actual_index].reshape(-1, 7)  # Reshape back to (time_steps, features)
    sample_len = sample.shape[0]
    sample_start = find_index(imu_time, collision_time[actual_index] - duration + offset)
    sample_end = sample_start + sample_len

    plt.figure(figsize=(7.4, 5.4))  # Double size to accommodate both plots side by side
    plt.subplot(3, 1, 1)
    plt.plot(imu_time[sample_start:sample_end], sample[:, 0], color=colors[0], linestyle='-', linewidth=1, label='X')
    plt.plot(imu_time[sample_start:sample_end], sample[:, 1], color=colors[1], linestyle='-', linewidth=1, label='Y')
    plt.plot(imu_time[sample_start:sample_end], sample[:, 2], color=colors[2], linestyle='-', linewidth=1, label='Z')
    plt.legend(['X', 'Y', 'Z'], frameon=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Linear acceleration (m/s2)')
    plt.title(f'Linear Acc. - True: {true_label}, Predicted: {predicted_label}')

    plt.subplot(3, 1, 2)
    plt.plot(imu_time[sample_start:sample_end], sample[:, 3], color=colors[0], linestyle='-', linewidth=1, label='X')
    plt.plot(imu_time[sample_start:sample_end], sample[:, 4], color=colors[1], linestyle='-', linewidth=1, label='Y')
    plt.plot(imu_time[sample_start:sample_end], sample[:, 5], color=colors[2], linestyle='-', linewidth=1, label='Z')
    plt.legend(['X', 'Y', 'Z'], frameon=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title('Angular Velocity')

    plt.subplot(3, 1, 3)
    plt.plot(imu_time[sample_start:sample_end], sample[:, 6], color='black', linestyle='-', linewidth=1, label='Thrust')
    plt.legend(['Thrust'], frameon=False)
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust')
    plt.title('Thrust')

    plt.tight_layout()

plt.show()