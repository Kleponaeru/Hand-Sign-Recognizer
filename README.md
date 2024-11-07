# Hand Sign Recognizer

Hand Sign Recognizer is a machine learning-based project that detects and interprets American Sign Language (ASL) hand signs from real-time video input. By leveraging a Convolutional Neural Network (CNN) model, this project can recognize gestures and display the corresponding letter in real time.

## Features

- Real-Time Recognition from a live webcam feed
- Displays the predicted sign on the video stream
- Easily trainable on additional signs or symbols if needed

## Tech Stack

- Python
- TensorFlow for model training and prediction
- OpenCV for capturing and processing real-time video
- NumPy for data manipulation

## Dataset 
Source: www.Kaggle.com

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-sign-recognizer.git
   cd hand-sign-recognizer
2. Install the requirements:
   ```bash
   pip install tensorflow opencv-python numpy
3. Run the project:
   ```bash
   python recognize_gesture.py
