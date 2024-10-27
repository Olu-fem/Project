from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from unet_model import unet
from alexnet_model import alexnet
from resnet_model import resnet50
import numpy as np
from file import images, labels

# Normalize the image data
images = np.array(images) / 255.0

# Ensure that labels are NumPy arrays
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range=0.2, 
    horizontal_flip=True
)
datagen.fit(X_train)

# Training U-Net
unet_model = unet()
unet_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Training AlexNet
alexnet_model = alexnet()
alexnet_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Training ResNet
resnet_model = resnet50()
resnet_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
