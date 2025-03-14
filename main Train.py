 
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array

 
# Define dataset path
image_directory = 'datasets/'
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

 

# Preprocess dataset
dataset = []
label = []
INPUT_SIZE = 224  # Adjusted for transfer learning compatibility

def load_images(image_list, label_value, folder):
    for image_name in image_list:
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_directory, folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
                dataset.append(img_to_array(image))
                label.append(label_value)

 

# Load images
load_images(no_tumor_images, 0, 'no')
load_images(yes_tumor_images, 1, 'yes')

 

# Convert to NumPy arrays
dataset = np.array(dataset, dtype=np.float32) / 255.0  # Normalize
label = np.array(label)

 

# Split data
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
 


# Build CNN model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))
cnn_model.save('BrainTumor_CNN.h5')

# Build Transfer Learning Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

transfer_model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
transfer_model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))
transfer_model.save('BrainTumor_TransferLearning.h5')
