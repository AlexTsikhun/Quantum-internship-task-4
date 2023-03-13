import os
import rasterio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import random

import tensorflow as tf

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

from keras.utils import normalize

import matplotlib.pyplot as plt
import segmentation_models as sm

from tensorflow.python.keras.models import load_model


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


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

#Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

image_dataset = np.squeeze(image_dataset, axis=3)
# mask_dataset = np.squeeze(mask_dataset, axis=3)
print(image_dataset.shape, mask_dataset.shape)


X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.1, random_state = 0)

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]


model = load_model('models/unet-sm.h5', compile=False)


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
