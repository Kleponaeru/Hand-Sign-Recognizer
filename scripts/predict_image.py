import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model/sign_language_model.h5')

# Function to preprocess image and predict
def predict_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = image.reshape(-1, 64, 64, 1)  # Reshape for model

    # Predict
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return predicted_class

# Example usage
image_path = 'path_to_your_image.png'  # Provide the path to an image
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")
