import numpy as np

# Parameters
num_samples  = 100  # Number of samples
num_features = 7    # Number of features per sample
num_classes  = 4    # Number of classes

# Generate random data for features
data = np.random.randn(num_samples, num_features)

# Generate random labels
labels = np.random.randint(0, num_classes, num_samples)

# Save data to test_data.txt
np.savetxt('test_data.txt', data, delimiter=',')

# Save labels to label.txt
np.savetxt('label.txt', labels, fmt='%d', delimiter=',')

print("Test data and labels generated and saved successfully!")
