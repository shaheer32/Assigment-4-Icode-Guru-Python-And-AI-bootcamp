# Assigment-4-Icode-Guru-Python-And-AI-bootcamp


Project 1: Cat vs Dog CNN Classifier

Overview

A custom Convolutional Neural Network (CNN) built from scratch to classify images of cats and dogs using binary classification.

Features

Custom CNN architecture with 4 convolutional blocks

Data augmentation (rotation, width/height shift, zoom, horizontal flip)

Dropout layers for regularization

Early stopping to prevent overfitting

Model checkpointing (best model saved automatically)

Training & validation accuracy/loss visualization

Prediction function for new images

Model saving for deployment


Project 2: Flower Classification (Transfer Learning)

Overview

A multi-class image classification project using MobileNetV2 with transfer learning on the TensorFlow Flowers dataset.

Features

Pretrained MobileNetV2

Two-phase training strategy:

Feature extraction (frozen base model)

Fine-tuning (unfreezing selected layers)

Data augmentation

Learning rate reduction on plateau

Prediction visualization with class probabilities

Detailed evaluation metrics

 Key Technical Points
Automatically downloads TensorFlow Flowers dataset

Softmax activation for multi-class classification

Strategic layer freezing/unfreezing

Input image size: 224 × 224

Optimized for accuracy with minimal training time



Project 3: Spam Detection (NLP)

Overview

An end-to-end spam detection system using classical NLP techniques and machine learning models, designed for real-time predictions and deployment.

Features

Complete text preprocessing pipeline

Two vectorization methods:

Bag of Words (BoW)

TF-IDF

Two machine learning models:

Naive Bayes

Logistic Regression

Side-by-side comparison of all 4 model combinations

Confusion matrix & detailed performance metrics

Real-time prediction function

Model persistence using pickle

Key Technical Points

Compares 4 configurations:

NB + BoW

NB + TF-IDF

LR + BoW

LR + TF-IDF (Best Performing)

Designed for production deployment

Lightweight & fast inference
