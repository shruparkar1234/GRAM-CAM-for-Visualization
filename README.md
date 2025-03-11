# Activation Heatmap and CNN Model for Image Classification

## Overview

This repository provides an implementation of a Convolutional Neural Network (CNN) for image classification, including preprocessing steps and a function to generate activation heatmaps using Grad-CAM. The model is trained using TensorFlow and Keras on a dataset of chest X-ray images.

## Requirements

Ensure you have the following dependencies installed before running the scripts:
```bash
pip install numpy pandas matplotlib tensorflow pillow
```
##  Dataset

The dataset consists of chest X-ray images stored in the following directory structure:

/content/drive/MyDrive/Colab Notebooks/chest_xray_multiclass
│── train
│── test

## Code Explanation

Importing Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping

## Loading and Preprocessing the Data
```bash
train_dir = "/content/drive/MyDrive/Colab Notebooks/chest_xray_multiclass/train"
test_dir = "/content/drive/MyDrive/Colab Notebooks/chest_xray_multiclass/test"

training_data = image_dataset_from_directory(
    train_dir, labels='inferred', label_mode='int', batch_size=32, image_size=(224,224))

test_data = image_dataset_from_directory(
    test_dir, labels='inferred', label_mode='int', batch_size=32, image_size=(224,224))
```
## Converting Dataset to NumPy Arrays
```bash
def dataset_to_numpy(dataset):
    images, labels = [], []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

X_train, y_train = dataset_to_numpy(training_data)
X_test, y_test = dataset_to_numpy(test_data)
```
## Normalizing the Data
```bash
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32') / 255

Building the CNN Model

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:], name="firstConvLayer"),
    layers.MaxPooling2D((2,2), name="firstMaxPoolLayer"),
    layers.Conv2D(64, (3, 3), activation='relu', name="secondConvLayer"),
    layers.MaxPooling2D((2,2), name="secondMaxPoolLayer"),
    layers.Flatten(name="flatteningLayer"),
    layers.Dense(64, activation='relu', name="firstDenseLayer"),
    layers.Dense(3, activation='softmax', name="predictionLayer")
])

model.summary()
```
## Generating Activation Heatmap using Grad-CAM
```bash
def make_activation_heatmap(model, img_array, label, last_conv_layer_name, classifier_layer_names):
    pred = model.predict(img_array)
    print(f"Model's prediction : {np.argmax(pred)}")
    print(f"Actual label : {label}")
    
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    plt.matshow(heatmap)
    img = np.squeeze(img_array, axis=0)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap).resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    
    superimposed_img = keras.utils.array_to_img(jet_heatmap * 0.4 + img)
    plt.imshow(superimposed_img)
    plt.show()
```

## Running the Model
- Load and preprocess the dataset.
- Train the CNN model using model.fit().
- Use make_activation_heatmap() to visualize the model's decision-making process.
