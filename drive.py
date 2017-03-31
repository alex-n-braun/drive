import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

from Database import normImg


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

import locale

@sio.on('telemetry')
def telemetry(sid, data):
    #print('Enter telemetry(sid, data)')
    # The current steering angle of the car
    old_steering_angle = locale.atof(data["steering_angle"])/25.0
    # The current throttle of the car
    throttle = data["throttle"]
    print('old: ', old_steering_angle, data["steering_angle"], throttle)
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = normImg(np.asarray(image))
    #print(image_array)
    #print('Image transformed to numpy array.')
    transformed_image_array = image_array[None, :, :, :]
    #print('Image normalized.')
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    # print(steering_angle, throttle)
    #old_steering_angle=wangle+steering_angle)
    #print(old_steering_angle)
    
    # rescale normalized steering angle back to a value range -1.0 ... 1.0
    f=1.0/0.9
    send_control(steering_angle*f, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    s=steering_angle
    print('new: ', s, throttle)
    sio.emit("steer", data={
    'steering_angle': s.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

from keras.optimizers import Adam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())

    print('Model ', args.model, ' loaded.')

    # model.compile("adam", "mse")
    adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(adam, 'mean_squared_error', ['accuracy'])
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    print('Weights ', weights_file, ' loaded.')
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)