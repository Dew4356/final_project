import os
import shutil
import random

# Set your dataset folder path and desired output folder paths
data_dir = '../dataset'
train_dir = '../train_set_5'
val_dir = '../valid_set_5'
test_dir = '../test_set_5'

# Create the output folders if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files in the dataset directory
all_files = os.listdir(data_dir)
all_files = [file for file in all_files if file.endswith(('jpg', 'jpeg', 'png'))]  # Filter out non-image files

# Shuffle the files randomly
random.seed(42)  # Set a seed for reproducibility
random.shuffle(all_files)

# Calculate the split indices
split_idx1 = int(len(all_files) * 0.7)
split_idx2 = int(len(all_files) * (0.7 + 0.15))

# Split the files into train, validation, and test sets
train_files = all_files[:split_idx1]
val_files = all_files[split_idx1:split_idx2]
test_files = all_files[split_idx2:]

# Copy the train, validation, and test files to the corresponding output folders
for file in train_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))
