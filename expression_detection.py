import numpy as np
import cv2
import tensorflow as tf

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameter as hp
from models import YourModel, VGGModel, MobileNetModel, ResNetModel , EfficientNet
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

## Self-Designed Model
model = YourModel()
load_checkpoint = os.path.abspath('checkpoints/your_model/050823-053020/your.weights.e049-acc0.6002.h5') ## unzip your.weights.e049-acc0.6002.h5 before you run this code

## Uncomment to try out pretrained model

## MobileNet Model
#model = MobileNetModel()
#load_checkpoint = os.path.abspath('checkpoints/mobilenet_model/050723-235439/mobilenet.weights.e026-acc0.6956.h5')

## VGG Model
#model = VGGModel()
#load_checkpoint = os.path.abspath("checkpoints/vgg_model/050823-025153/vgg.weights.e028-acc0.6571.h5")

## ResNet Model
#model = ResNetModel()
#load_checkpoint = os.path.abspath("checkpoints/resnet_model/050823-190449/resnet.weights.e029-acc0.7075.h5")
model(tf.keras.Input(shape=(48, 48, 3)))
model.load_weights(load_checkpoint, by_name=False)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Expression labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

frame_counter = 0
expression = ""

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_crop = gray[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48)) # revise (48,48) to (224,224) if you use pretrained mode such as vgg
        face_crop = face_crop / 255.0
        face_crop = np.expand_dims(face_crop, axis=(0, -1))

        # Predict the expression every 24 frames
        if frame_counter % 24 == 0:
            
            print(face_crop.shape)
            f = cv2.merge((face_crop,face_crop,face_crop))
            
            f = np.reshape(f,(-1,48,48,3)) # revise (-1,48,48,3) to (-1,224,224,3) if you use pretrained mode such as vgg
            print(f.shape)
            prediction = model.predict(f)
            
            max_index = int(np.argmax(prediction))
            expression = labels[max_index]

        # Display the expression
        cv2.putText(frame, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Real-time Face Detection and Expression Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Close webcam and windows
cap.release()
cv2.destroyAllWindows()