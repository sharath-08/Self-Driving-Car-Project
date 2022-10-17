import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

def getName(filePath):
    return filePath.split('\\')[-1]
    # returns the last path name in a path-list. Removes the '\' in the line

def importDataInfo(path):
    columns = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv (os.path.join(path,'driving_log.csv'),names = columns)
    # print(data.head())
    # print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())
    print('Total Images Imported:', data.shape[0])
    return data


def BalanceData(data, display = True):
    nBins = 31
    samplesPerBin = 2500
    hist, bins = np.histogram(data['Steering'],nBins)
    # print(bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    # print(center)
    plt.bar(center,hist,width =0.06)
    plt.plot((-1,1),(samplesPerBin,samplesPerBin))
    plt.show()

    # removeIndexList stores values to be deleted
    removeIndexList = []
    # looping through each bin-value. In our example its 31 bins, with different values
    for j in range(nBins):

        #binDataList stores values from data['steering'] that belong to a certain bin
        binDataList = []

        #looping through each value in our data['Steering'] column and filtering which bin it goes into
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                #once the bin is found, we store it in
                binDataList.append(i)
        #we shuffle our binDataList so that when we delete, we dont just delete in order, thus deleting only the smaller values
        binDataList = shuffle(binDataList)
        #stores data samples from 2500 and up to be deleted
        binDataList = binDataList[samplesPerBin:]
        #appends the data to be deleted in removeIndexList
        removeIndexList.extend(binDataList)
    print('Removed Images: ',len(removeIndexList))
    #dropping the data from removeIndexList from the main data sample
    data.drop(data.index[removeIndexList],inplace = True)
    print('Remaining Images: ',len(data))

    if display:
        hist,_ = np.histogram(data["Steering"], nBins)
        plt.bar(center, hist,width = 0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    return data



def loadData(path,data):
    imagesPath = []
    steering = []

    for i in range (len(data)):
        indexData = data.iloc[i]
        # print(indexData)
        imagesPath.append(os.path.join(path,'IMG',indexData[0]))
        steering.append(float(indexData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def augumentImage(imgPath, steering):
    #import the image
    img = mpimg.imread(imgPath)

    #we start augumenting the image using Pan, zoom, brightness and flip.
    #Affine function- allows us to scale values independently per axis

    ## We use a randomiser to augment the data in random ways. If the generated value is greater than 50% we apply the aug
    ## Note we can apply multiple augmentations on one image

    ##PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    ##ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    ##BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    ##FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        #if we flip the image, we must flip the steering angle
        steering = -steering

    return img,steering

#imgRe , st = augumentImage('test.jpg',0)
# plt.imshow(imgRe)
# plt.show()


def preProcessing (img):

    ##Cropping image to only have road, and to remove mountain and other scenery
    img = img[60:135,:,:]

    ##Changing our colorspace from RGB to YUV to make the road lines more visible
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    #adding some blur to image
    img = cv2.GaussianBlur(img,(3,3),0)

    ##Resizing our image from 200 to 66
    img = cv2.resize(img,(200,66))

    ##data normalisation- arranging our values from 0 to 1
    img = img/255

    return img

# imgRe = preProcessing(mpimg.imread('test.jpg'))
# plt.imshow(imgRe)
# plt.show()

##This function picks images at random to augment and preprocess before sending to our model as a batch.
##May generate multiple batches of images picked at random
def batchGen(imagesPath, steeringList, batchSize,trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            ##Picking a random image and storing its index
            index = random.randint(0,len(imagesPath)-1)

            ##Since we don't want to augment our validation image. We use the train-flag to check if image is
            ##used for validation
            if(trainFlag == True):
                img, steering = augumentImage(imagesPath[index],steeringList[index])

            ##If image is used for validation, we just read image and send it without augmentation
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))


def createModel():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')
    return model