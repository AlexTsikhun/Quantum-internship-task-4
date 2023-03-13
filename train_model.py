import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D

import matplotlib.pyplot as plt


# # own model
# def unet(input_size):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
#     conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
#     pool1 = MaxPooling2D((2, 2))(conv1)

#     conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
#     conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
#     pool2 = MaxPooling2D((2, 2))(conv2)

#     conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
#     conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
#     pool3 = MaxPooling2D((2, 2))(conv3)

#     conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
#     conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D((2, 2))(drop4)

#     conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
#     conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv5)
#     drop5 = Dropout(0.5)(conv5)

#     up6 = Conv2D(512, (2, 2), activation="relu", padding="same")(UpSampling2D((2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(merge6)
#     conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv6)

#     up7 = Conv2D(256, (2, 2), activation="relu", padding="same")(UpSampling2D((2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(merge7)
#     conv7 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv7)

#     up8 = Conv2D(128, (2, 2), activation="relu", padding="same")(UpSampling2D((2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(merge8)
#     conv8 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv8)

#     up9 = Conv2D(64, (2, 2), activation="relu", padding="same")(UpSampling2D((2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(merge9)
#     conv9 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv9)
#     conv9 = Conv2D(2, (3, 3), activation="relu", padding="same")(conv9)
#     conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

#     return model

# raster_path = "data/T36UXV_20200406T083559_TCI_10m.jp2"

# with rasterio.open(raster_path) as src:
#     width = src.width
#     height = src.height
#     meta = src.meta


patch_size = 256

# if set original size - 10980, will get error in 105 line of code
width = height = 10752

raster_images = []
mask_images = []
for i in range(0, width, patch_size):
    for j in range(0, height, patch_size):
        raster_img = cv2.imread(f"data/patches/images/image_{i}_{j}.jp2", cv2.IMREAD_COLOR)
        raster_img = cv2.cvtColor(raster_img, cv2.COLOR_RGB2BGR)
        raster_images.append(raster_img)
#         print(raster_images)

        mask_img = cv2.imread(f"data/patches/masks/mask_{i}_{j}.jp2", 0)
#         print(f"/kaggle/input/patches-images-split/mask_{i}_{j}.jp2")
#         break
        mask_images.append(mask_img)
    


# convert list to array (NN accepts array!)
raster_images = np.array(raster_images)
mask_images = np.array(mask_images)

X = raster_images
Y = mask_images
Y = np.expand_dims(Y, axis=3)
print(X.shape)
print(Y.shape)

X = X / 255.0
Y = Y / 255.0

print(X.shape, Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=123)

model = unet((256, 256, 3))

model_path = "unet.h5"
checkpoint = ModelCheckpoint(model_path,
                             monitor="val_loss",
                             mode="min",
                             verbose=1)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print('fit')
history = model.fit(X_train, y_train, epochs=10, batch_size=2, callbacks=[checkpoint])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
