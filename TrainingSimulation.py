print('Setting up')
import os
os.environ['TF_CPP_MIN_KOG_LEVEL'] = '3'
from utils import *
from sklearn.model_selection import train_test_split
import socketio
import eventlet
from flask import Flask


#### STEP 1: data collation
path = 'myData'
data = importDataInfo(path)

#### STEP 2: data visualisation and balancing
data = BalanceData(data)

#### STEP 3: seperating the data from a pandas dataframe to seperate lists
imagesPath, steering = loadData(path,data)
print(imagesPath[0],steering[0])


#### STEP 4: data validation. Tests our model data to our validation data.
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steering,test_size = 0.2, random_state=5)
print('Total training images: ',len(xTrain))
print('Total validation images: ',len(xVal))

####STEP 5:AUGMENTING THE IMAGE


#### STEP 6: PRE-PROCESSING THE IMAGE

####STEP 7:


####STEP 8: Create Model
model = createModel()
model.summary()


####STEP 9: Training our model.

##We use tensor fit() function, with training data sent as batches, and with the validation data sent as seperate batch
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)


####STEP 10: Write our model to file
model.save('model.h5')
print("model saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()