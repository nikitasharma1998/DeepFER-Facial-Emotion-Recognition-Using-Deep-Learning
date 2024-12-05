# DeepFER: Facial Emotion Recognition Using Deep Learning
## Introduction
Welcome to DeepFER, a project focused on developing a robust system for recognizing emotions from facial expressions using deep learning techniques. This repository contains the code, models, and resources used to build and train a Convolutional Neural Network (CNN) for accurately classifying emotions such as happiness, sadness, anger, surprise, and more from facial images.

### Key Features
Utilizes state-of-the-art deep learning architectures for facial emotion recognition.
Includes data augmentation techniques to enhance model generalization.
Integrates Transfer Learning for leveraging pre-trained models.
Real-time emotion recognition capabilities for live video feeds.
Application development for interactive interfaces and human-computer interaction.

### Contents

experiments/: Directory with notebooks of model architectures. Use Final_FER for final version.

app/: Application development files for integrating emotion recognition into user interfaces.

README.md: Project overview and documentation

### Functionality
1. Use the final saved model in app.py
2. Runs a streamlit app which can take input either from webcam(Real-time) or image uploads.

### Dataset
The dataset used in this project consists of facial images categorized into seven emotion classes: angry, sad, happy, fear, neutral, disgust, and surprise. Each image is labeled with one of these emotions, enabling supervised learning for emotion classification.

## Data Preprocessing
### Data Augmentation
To enhance model generalization, various data augmentation techniques are applied during preprocessing. These include:
- Rotation
- Horizontal and vertical flipping
- Zooming
- Width and height shifts

### Data Normalization
Pixel values of images are normalized to a range of [0, 1]. This normalization simplifies model convergence during training by ensuring that all input values are within a consistent range.

## Model Architecture
The model architecture is based on ResNet50V2, a deep convolutional neural network pre-trained on the ImageNet dataset. The architecture includes:
- Base Model: ResNet50V2 (pre-trained on ImageNet)
- Additional Layers:
  - Dropout (0.25): Regularization technique to prevent overfitting
  - BatchNormalization: Normalizes the activations of the previous layer at each batch
  - Flatten: Converts the 2D matrix of features into a vector
  - Dense (64, activation='relu'): Fully connected layer with ReLU activation
  - BatchNormalization: Normalizes the activations of the previous layer at each batch
  - Dropout (0.5): Another dropout layer for further regularization
  - Dense (7, activation='softmax'): Output layer with softmax activation for multi-class classification (seven emotions)

## Training
### Training Setup
The model is compiled with:
- Optimizer: Adam optimizer with default parameters
- Loss Function: Categorical Cross-Entropy, suitable for multi-class classification tasks
- Metrics: Accuracy, to monitor model performance during training

### Callbacks
Callbacks are used to monitor the training process and adjust learning dynamics:
- ModelCheckpoint: Saves the best model based on validation accuracy
- EarlyStopping: Stops training early if validation accuracy doesn't improve for a set number of epochs
- ReduceLROnPlateau: Reduces learning rate if validation loss plateaus, to fine-tune model performance

### Training Process
The model is trained over multiple epochs with batch processing. The dataset is split into training and validation sets to evaluate model performance.

## Evaluation
### Model Performance
After 15 epochs of training, the model achieved a test accuracy of 67%. Training and validation curves showed convergence after the 9th epoch, indicating effective learning and generalization capabilities.

## Conclusion
DeepFER demonstrates effective emotion recognition capabilities using state-of-the-art deep learning techniques. The model can be further optimized and deployed in real-world applications where accurate emotion detection from facial expressions is essential.

