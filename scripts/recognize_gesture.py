import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model/sign_language_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))  # Resize to 28x28 (model's input size)
    gray = np.reshape(gray, (1, 28, 28, 1)) / 255.0  # Normalize and reshape to (1, 28, 28, 1)

    # Make a prediction
    prediction = model.predict(gray)
    predicted_label = np.argmax(prediction)

    # Display the result on the frame
    cv2.putText(frame, f"Predicted: {chr(predicted_label + 65)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
