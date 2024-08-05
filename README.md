# Moon Phase Prediction

This repository contains code for a deep learning model that predicts the phase of the moon from an input image. The model is built using Convolutional Neural Networks (CNNs) and classifies images into one of the eight moon phases.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Making Predictions](#Making-Predictions)
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
The dataset consists of images of the moon labeled with their respective phases. The images are organized in subdirectories named after the phases. Ensure that the dataset is unzipped and placed in the correct directory structure.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with ReLU activation
- Dropout layer
- Output layer with softmax activation

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

model.summary()
```
![model](https://github.com/user-attachments/assets/c2e99836-7930-4790-a0cf-5ad0943010b5)


## Training
The images are loaded, resized, and normalized. The labels are converted to categorical format. The dataset is split into training and testing sets. The model is then compiled and trained.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
```
![epochs](https://github.com/user-attachments/assets/b98d36e6-4fda-45fe-b7bb-b33a40b57a09)


## Evaluation
The model's performance is evaluated on the test set.

```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

![acc](https://github.com/user-attachments/assets/207b1e21-a604-448d-bf52-a180e50fee89)


## Making Predictions
You can use the trained model to predict the moon phase of new images.

```python
def predict_moon_phase(model, img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img_reshaped)
    return CATEGORIES[np.argmax(prediction)]
```

Testing model on different images

```python
img_path = 'test.jpg'
predicted_phase = predict_moon_phase(model, img_path)
print("The predicted moon phase is:", predicted_phase)
```
![test_result](https://github.com/user-attachments/assets/48208ee7-c213-4d84-8895-6b5b98c5f88f)

```python
img_path = 'test1.jpg'
predicted_phase = predict_moon_phase(model, img_path)
print("The predicted moon phase is:", predicted_phase)
```
![test1_result](https://github.com/user-attachments/assets/bd716618-eb62-4a5e-89b7-e26320af27a7)


## Results
The training and validation loss and accuracy are plotted to visualize the model's performance over epochs.

![accuracy](https://github.com/user-attachments/assets/3fbab1fe-3327-48e7-b9ca-501172fc552b)


![loss](https://github.com/user-attachments/assets/7a89291e-d8fd-458a-942e-d12f33c46214)
