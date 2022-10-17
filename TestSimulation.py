print('Setting up')
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__) #'__main__'
maxSpeed = 5

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


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcessing(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0,0)

def sendControl(steering, throttle):
    sio.emit('steer',data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)