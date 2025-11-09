import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, Model, Input
from tensorflow import keras


print("TF", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Set Matplotlib defaults
plt.rc(group='figure', autolayout=True)
plt.rc(group='axes',
       labelweight='bold',
       titleweight='bold',
       labelsize='large',
       titlesize=18,
       titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings('ignore') # clean up output cells

# Load training and validation sets
file_path = 'data/'

ds_train_ = image_dataset_from_directory(
    file_path,
    labels='inferred',
    class_names=['human','anime'],
    label_mode='binary',  # Only for exactly 2 classes; change to 'categorical' if more
    validation_split=0.2,
    subset='training',
    seed=31415,
    image_size=[299,299],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    file_path,
    class_names=['human','anime'],
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=31415,
    image_size=[299,299],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

print(ds_train_.class_names)

# Data Pipeline
def convert_to_float(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Fitting the data into the train model
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
print('Setup complete')

# Load InceptionV4 using Tensorflow Keras
# from keras_inceptionV4.inception_v4 import inception_v4
#
# base_model = inception_v4(
#     include_top=False,
#     weights='imagenet',
#     dropout_keep_prob=0.5,
#     num_classes=1000
# )

# === Preprocessing (use InceptionV3 preprocessing) ===
preprocess_input = tf.keras.applications.inception_v3.preprocess_input

def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)  # scales to [-1,1] as Inception expects
    return images, labels

# Base model
base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(299, 299, 3)
)
base_model.trainable = False

# Optional data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.06),
], name="data_augmentation")

inputs = keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)           # optional
x = base_model(x, training=False)       # keep BN in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
loss = "binary_crossentropy"
metrics = ["accuracy"]

model = keras.Model(inputs, outputs)
# Compile and train head
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=loss,
    metrics=metrics,
)

callbacks = [
    keras.callbacks.ModelCheckpoint("inceptionv3_head.h5", save_best_only=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
]

print(model.summary())
print("model setup complete")


# Fit model

model.fit(ds_train, validation_data=ds_valid, epochs=5, callbacks=callbacks)


model.save("binary_character_identifier.h5")











