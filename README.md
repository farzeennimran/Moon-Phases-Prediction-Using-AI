# Moon Phase Prediction

This repository contains code for a deep learning model that predicts the phase of the moon from an input image. The model is built using Convolutional Neural Networks (CNNs) and classifies images into one of the eight moon phases.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Overview
The goal of this project is to accurately classify images of the moon into one of eight phases:
1. New Moon
2. Waxing Crescent
3. First Quarter
4. Waxing Gibbous
5. Full Moon
6. Waning Gibbous
7. Last Quarter
8. Waning Crescent

## Dataset
The dataset consists of images of the moon categorized into eight phases of moon. The images are loaded and preprocessed using TensorFlow's image dataset utility.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with ReLU and softmax activation for classification

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
```
![model](https://github.com/user-attachments/assets/c2e99836-7930-4790-a0cf-5ad0943010b5)


## Training
The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss. The dataset is split into training, validation, and test sets with a ratio of 70%, 20%, and 10% respectively. Training is performed over 11 epochs.

```python
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
```
![epochs](https://github.com/user-attachments/assets/b98d36e6-4fda-45fe-b7bb-b33a40b57a09)


## Evaluation
The models accuracy is around 0.69 

```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

![acc](https://github.com/user-attachments/assets/207b1e21-a604-448d-bf52-a180e50fee89)


## Usage
To use the model for predicting the moon phase of an input image, follow these steps:

1. Load and preprocess the image.
2. Use the trained model to predict the phase.
3. Map the predicted class to the corresponding moon phase.

```python
img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resize = tf.image.resize(img, (256,256))
resize_expanded = np.expand_dims(resize/255, 0)
test_prediction = model.predict(resize_expanded)
moon_phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
predicted_class = np.argmax(test_prediction, axis=1)[0]
print(f'Predicted moon phase is {moon_phases[predicted_class]}')
```

## Results
The training and validation loss and accuracy are plotted to visualize the model's performance over epochs.
![accuracy](https://github.com/user-attachments/assets/3fbab1fe-3327-48e7-b9ca-501172fc552b)


![loss](https://github.com/user-attachments/assets/7a89291e-d8fd-458a-942e-d12f33c46214)




