import numpy as np
import torch

# Function to load data from a text file
def load_data(file_path, delimiter=','):
    data = np.loadtxt(file_path, delimiter=delimiter)
    return data

# Load test data and labels
test_data = load_data('test_data.txt')
labels = load_data('label.txt', delimiter=',')

# Convert to PyTorch tensors
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)  # Assuming labels are integers

print("Test data shape:", test_data_tensor.shape)
print("Labels shape:", labels_tensor.shape)

# Example usage in a DataLoader
dataset = torch.utils.data.TensorDataset(test_data_tensor, labels_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Verify the data loader
for inputs, labels in data_loader:
    print("Batch of inputs shape:", inputs.shape)
    print("Batch of labels shape:", labels.shape)
    break  # Print only the first batch for verification
