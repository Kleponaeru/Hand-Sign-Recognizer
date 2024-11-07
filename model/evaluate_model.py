import tensorflow as tf
import numpy as np

# Load test data and model
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')
model = tf.keras.models.load_model('sign_language_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
