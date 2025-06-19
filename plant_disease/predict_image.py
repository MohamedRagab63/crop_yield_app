import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("best_model.keras")

# Load class names (according to the order in train_generator.class_indices)
import json

# Load class names from file
with open("class_names.json") as f:
    class_indices = json.load(f)

# Sort classes by index to maintain correct order
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

# Function to prepare the image
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Same preprocessing as during training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Image path for prediction (change the path to your image)
img_path = "test_leaf.jpg"

# Process the image
processed_img = prepare_image(img_path)

# Make prediction
prediction = model.predict(processed_img)
predicted_class = class_names[np.argmax(prediction)]

# Print the result
print(f"The image was classified as: {predicted_class}")

# Display the image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()
