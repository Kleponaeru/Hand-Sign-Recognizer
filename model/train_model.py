import numpy as np
from build_model import build_model

# Load preprocessed data
X_train = np.load('train_images.npy')
y_train = np.load('train_labels.npy')
X_val = np.load('val_images.npy')
y_val = np.load('val_labels.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_labels.npy')

# Build the model
model = build_model()

# Train the model with the training data and validate with the validation data
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Save the trained model
model.save('model/sign_language_model.h5')

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
