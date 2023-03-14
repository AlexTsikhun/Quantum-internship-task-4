<<<<<<< HEAD
import os
import rasterio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import random

from tensorflow import keras

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from keras.models import Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

from keras.utils import normalize

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import matplotlib.pyplot as plt
import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

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


image_directory = 'data/patches/images/'
mask_directory = 'data/patches/masks/'

SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jp2'):
#         print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, cv2.IMREAD_COLOR)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        # convert list to array (NN accepts array!)
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
#     print(i)
    if (image_name.split('.')[1] == 'jp2'):
        image = cv2.imread(mask_directory+image_name, 0)
#         print(mask_directory+image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        # convert list to array (NN accepts array!)
        mask_dataset.append(np.array(image))

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)
# mask_dataset = np.expand_dims(mask_dataset, axis=3)

#Normalize images
image_dataset = normalize(image_dataset, axis=1)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((mask_dataset),3) /255.

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.1, random_state = 123)

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]


model = sm.Unet(BACKBONE, encoder_weights='imagenet') # classes=2, activation='softmax'
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

history = model.fit(X_train, y_train, 
                    batch_size = 2, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('models/unet-sm_5.h5')

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
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


test_img_number = 179#random.randint(0, len(X_test))
# print(test_img_number)
test_img = X_test[test_img_number]
plt.imshow(test_img, cmap='gray')

test_img = test_img.astype('uint8')
test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
# plt.imshow(test_img, cmap='gray')
# test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)[0]


plt.imshow(y_train[test_img_number], cmap='gray')

prediction_image = prediction.reshape((256, 256))
plt.imshow(prediction_image)

# compare
fig, ax = plt.subplots(1, 2)
ax[0].imshow(mask_dataset[test_img_number])
ax[1].imshow(prediction_image)
plt.show()

# more visualize
test_img = cv2.imread('data/patches/images/image_0_0.jp2', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

test_img1 = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img1)[0]

#View and Save segmented image
prediction_image = prediction.reshape(256, 256)
# plt.imshow(prediction_image, cmap='gray')
# plt.imsave('membrane/test0_segmented.jpg', prediction_image, cmap='gray')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(test_img)
ax[1].imshow(prediction_image)
plt.show()
=======
import os
import rasterio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import random

## error with this code
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 
import tensorflow as tf

# from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

from keras.utils import normalize

import matplotlib.pyplot as plt
import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

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


image_directory = 'data/patches/images/'
mask_directory = 'data/patches/masks/'

SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jp2'):
#         print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, cv2.IMREAD_COLOR)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        # convert list to array (NN accepts array!)
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
#     print(i)
    if (image_name.split('.')[1] == 'jp2'):
        image = cv2.imread(mask_directory+image_name, 0)
#         print(mask_directory+image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        # convert list to array (NN accepts array!)
        mask_dataset.append(np.array(image))

image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
# Rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

image_dataset = np.squeeze(image_dataset, axis=3)
# mask_dataset = np.squeeze(mask_dataset, axis=3)
print(image_dataset.shape, mask_dataset.shape)


X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.1, random_state = 123)

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]


model = sm.Unet(BACKBONE, encoder_weights='imagenet') # classes=2, activation='softmax'
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('models/unet-sm.h5')

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
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


test_img_number = 179#random.randint(0, len(X_test))
# print(test_img_number)
test_img = X_test[test_img_number]
plt.imshow(test_img, cmap='gray')

test_img = test_img.astype('uint8')
test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
# plt.imshow(test_img, cmap='gray')
# test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)[0,:,:,0] > 0.05


plt.imshow(y_train[test_img_number], cmap='gray')

prediction_image = prediction.reshape((256, 256))
plt.imshow(prediction_image)


# more visualize
test_img = cv2.imread('data/patches/images/image_0_0.jp2', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

test_img1 = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img1)[0]

#View and Save segmented image
prediction_image = prediction.reshape(256, 256)
# plt.imshow(prediction_image, cmap='gray')
# plt.imsave('membrane/test0_segmented.jpg', prediction_image, cmap='gray')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(test_img)
ax[1].imshow(prediction_image)
plt.show()
>>>>>>> 9b98eb670ab240c4605ab22fcb118df735ee3cd4
