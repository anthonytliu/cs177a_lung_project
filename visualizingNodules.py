# Visualizing:
#
# original axial images
# masks generated from annotations of images
# lung mask generated from thesholding
# images used for training
# masks applied to training images
# images used for testing
# masks applied to testing images
# trained model's prediction segmentation of test images
#

import numpy as np
import matplotlib.pyplot as plt

# output_path should be where the train images, train masks, test images,
# test masks, and lung masks are extracted with the LUNA_mask_extraction.py
# and LUNA_segment_lung_ROI.py scripts
output_path = "/Users/danisim/Desktop/ndsb_tutorial/"
imgs = np.load(output_path+'images_0023_0144.npy') # original axial slice
masks = np.load(output_path+'masks_0023_0144.npy') # extracted mask
lung_mask = np.load(output_path+'lungmask_0023_0144.npy') # extracted lung mask
for i in range(len(imgs)):
    print( "image %d" % i)
    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(imgs[i],cmap='gray')
    ax[0,1].imshow(masks[i],cmap='gray')
    ax[1,0].imshow(imgs[i]*masks[i],cmap='gray')
    ax[1,1].imshow(lung_mask[i], cmap='gray')
    plt.show()
    input("hit enter to cont : ")

testimgs = np.load(output_path+'testImages.npy')
testmasks = np.load(output_path+'testMasks.npy')
trainimgs = np.load(output_path+'trainImages.npy')
trainmasks = np.load(output_path+'trainMasks.npy')

print(testimgs.shape)   #(67, 1, 512, 512)
print(testmasks.shape)  #(67, 1, 512, 512)
print(trainimgs.shape)  #(269, 1, 512, 512)
print(trainmasks.shape) #(269, 1, 512, 512)

# print(len(testimgs))
# print(len(testimgs[0]))
# print(len(testimgs[0][0]))
# print(len(testimgs[0][0][0]))

print('Example of Test Image')
plt.imshow(testimgs[0][0])

print('Extracted Test Mask overlaid on Test Image')
plt.imshow(testmasks[0][0]*testimgs[0][0])

print('Example of Training Image')
plt.imshow(trainimgs[0][0])

print('Extracted Training Mask overlaid on Training Image')
plt.imshow(trainmasks[0][0]*trainimgs[0][0])

print(imgs.shape)       #(10,512,512)
print(masks.shape)      #(10,512,512)
print(lung_mask.shape)  #(3,512,512)

from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = get_unet()

# load weights from a previously trained neural network
model.load_weights('/Users/danisim/Documents/GitHub/DSB3Tutorial/unet.hdf5')

# subset of four test images
testimgs[:-60].shape                #(7,1,512,512)

# plt.imshow(testimgsubset[0][0]*testoutputs[0][0])

imgs_test=testimgs[:-60]
imgs_mask_test_true=testmasks[:-60]
num_test = len(imgs_test)
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
for i in range(num_test):
    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
np.save('masksTestPredicted.npy', imgs_mask_test)
mean = 0.0
for i in range(num_test):
    mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
mean/=num_test
print("Mean Dice Coeff : ",mean) # Mean Dice Coeff :  2.7400186830570795e-05

testimgsubset=testimgs[:-63]
testmasks = testmasks[:-63]
masks_predicted = np.load('/Users/danisim/Documents/GitHub/DSB3Tutorial/My Tutorial Attempt/masksTestPredicted.npy')
print(masks_predicted.shape) #(7,1,512,512)
plt.imshow(masks_predicted[6][0])

testoutputs=model.predict(testimgsubset)

plt.imshow(testimgsubset[2][0], cmap='brg')
plt.imshow(testmasks[2][0], cmap='brg')
plt.imshow(testoutputs[2][0], cmap='brg')
plt.imshow(testimgsubset[2][0]-testoutputs[2][0], cmap='brg')

# generate figure 3 for final paper
for i in range(len(imgs_test)):
    print( "image %d" % i)
    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(imgs_test[i][0],cmap='gray')
    ax[0,1].imshow(imgs_mask_test_true[i][0],cmap='rainbow')
    ax[1,0].imshow(masks_predicted[i][0],cmap='rainbow')
    ax[1,1].imshow(masks_predicted[i][0]*imgs_mask_test_true[i][0],cmap='rainbow')
    plt.show()
    input("hit enter to cont : ")
