import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths for the data
data_path = "data-set/"
train_csv_file = "data-set/sign_mnist_train.csv"  # Modify path if necessary
test_csv_file = "data-set/sign_mnist_test.csv"    # Modify path if necessary

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    images = []
    labels = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Extract the label (gesture)
        label = row['label']
        
        # Get pixel values (columns pixel 1 to pixel 34)
        pixel_values = row[1:].values  # Extract pixel columns (all except the first column 'label')
        
        # Normalize pixel values between 0 and 1
        image = pixel_values / 255.0
        
        # Reshape the image to 34x34 (since we have 34 pixels)
        image = image.reshape(28, 28, 1)
        
        # Append the image and label to the lists
        images.append(image)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Load train and test data
train_images, train_labels = load_data(train_csv_file)
test_images, test_labels = load_data(test_csv_file)

# Split the training data into training and validation sets
random_seed = 42
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=random_seed)

# Save the preprocessed data for easy use later
np.save('train_images.npy', X_train)
np.save('train_labels.npy', y_train)
np.save('val_images.npy', X_val)
np.save('val_labels.npy', y_val)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
