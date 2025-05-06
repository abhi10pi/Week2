# 🌿 Plant Disease Detection using CNN

This project aims to detect various plant diseases using image classification techniques powered by a Convolutional Neural Network (CNN). The model is trained on a dataset consisting of healthy and diseased plant leaf images, enabling accurate and automated disease detection.

---

## 🧠 Overview

Early detection of plant diseases is crucial for ensuring good crop yields. This project utilizes deep learning techniques, particularly CNNs, to classify plant leaf images into their respective disease categories.

---

## 📁 Dataset

- The dataset contains images of healthy and diseased leaves.
- It is organized into **training**, **validation**, and **testing** directories.
- Images are categorized into folders by disease type for supervised learning.

> 📌 **Note**: The dataset is stored in Google Drive and is automatically mounted in the notebook for access.

---

## 🔧 Project Structure

### I. Data Preparation

- ✅ Mounted Google Drive to access the dataset.
- ✅ Extracted the dataset from a zip file.
- ✅ Imported required libraries: `TensorFlow`, `NumPy`, `Pandas`, `Matplotlib`, etc.
- ✅ Defined paths for training, validation, and testing datasets.
- ✅ Used `ImageDataGenerator` for:
  - Loading images in batches.
  - Resizing and normalizing.
  - Augmenting training data (rotations, shifts).
- ✅ Visualized sample images from the training set.

### II. Model Building

- ✅ Built a **Convolutional Neural Network (CNN)** using Keras `Sequential` API:
  - **Convolutional Layers** for feature extraction.
  - **MaxPooling Layers** for downsampling.
  - **Flatten Layer** to convert features into a 1D vector.
  - **Dense Layers** for classification logic.
  - **Output Layer** with softmax activation for multi-class classification.
- ✅ Printed model summary to view architecture and parameters.

### III. Next Steps

- 🔄 **Train** the model using the training set.
- ✅ **Evaluate** model performance using the validation set.
- 🔧 **Fine-tune** hyperparameters for better accuracy.
- 🧪 **Test** final model performance on unseen test data.

---


   `
