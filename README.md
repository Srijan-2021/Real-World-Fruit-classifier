# Real-World Fruit Classifier 🍎

A deep learning-based image classification system designed to recognize five fruit categories: **Apple, Banana, Mango, Orange, and Strawberry** using TensorFlow, Keras, and MobileNetV2.

Unlike conventional fruit classification projects that rely on clean, studio-like images, this project focuses on handling challenging real-world conditions, including cluttered backgrounds, varying lighting, different camera angles, partial occlusions, and inconsistent image quality.

## Project Overview

The system uses **MobileNetV2 with Transfer Learning** to build an efficient and accurate image classifier. Starting with a clean benchmark dataset, the model underwent four major development iterations (V1–V4) to address background bias and improve its ability to recognize fruits in diverse environments.

The final version (V4) achieves **96.7% validation accuracy** on a custom dataset containing over **10,000 real-world and heavily augmented images**.

## Key Features

- **Transfer Learning:** Uses MobileNetV2 as the backbone for efficient feature extraction and image classification.
- **Five-Class Classification:** Recognizes Apple, Banana, Mango, Orange, and Strawberry.
- **Real-World Dataset:** Incorporates diverse images featuring different backgrounds, lighting conditions, orientations, and environments.
- **Automated Data Cleaning:** Helps identify and remove unsuitable or problematic images to improve dataset quality.
- **Data Augmentation:** Introduces variations in training images to improve generalization and reduce overfitting.
- **Class Weighting:** Addresses class imbalance during model training.
- **Hard-Example Mining:** Focuses additional training attention on difficult examples and misclassifications.
- **Iterative Model Development:** Four model versions were developed and evaluated to progressively overcome background dependency and improve classification performance.

## Performance

| Metric | Result |
|---|---|
| Fruit Categories | 5 |
| Dataset Size | 10,000+ images |
| Model Architecture | MobileNetV2 |
| Final Version | V4 |
| Validation Accuracy | 96.7% |

## Technology Stack

- **Programming:** Python
- **Deep Learning:** TensorFlow, Keras
- **Architecture:** MobileNetV2
- **Deployment:** Streamlit, Hugging Face Spaces
- **Data Processing:** Image preprocessing, cleaning, and augmentation

## Deployment

The trained V4 model is deployed as an interactive web application using Streamlit on Hugging Face Spaces. Users can upload fruit images and receive real-time classification predictions through a simple interface.

## Objective

The primary objective is to develop a practical fruit recognition system that performs reliably beyond controlled benchmark datasets, demonstrating the importance of dataset diversity, iterative experimentation, and robust model generalization in real-world computer vision applications.