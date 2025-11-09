import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Load trained model
test_model = load_model('binary_character_identifier.h5')

# Directory containing test images
test_file_directory = 'test/'

# Collect filenames of all image files in test folder
image_files = [
    os.path.join(test_file_directory, fname)
    for fname in os.listdir(test_file_directory)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Check if folder has images
if not image_files:
    raise ValueError(f"No image files found in directory: {test_file_directory}")

# Load, preprocess, and stack images
test_images = []
for img_path in image_files:
    img = tf.keras.utils.load_img(img_path, target_size=(299, 299))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    test_images.append(img_array)

test_images = np.stack(test_images, axis=0)

# Predict
predictions = test_model.predict(test_images)
# For binary classification, model might output shape (n, 1)
if predictions.shape[1] == 1:
    predicted_classes = (predictions > 0.5).astype("int32").flatten()
else:
    predicted_classes = np.argmax(predictions, axis=1)

# Display predictions alongside filenames
for file, pred in zip(image_files, predicted_classes):
    l_classes = ['human','anime']
    print(f"{os.path.basename(file)} → Predicted class: {l_classes[pred]}")
