import cv2
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

prediction = []
labels = {0: 'Safe driving', 1: 'Texting - right', 2: 'Talking on the phone - right', 3: 'Texting - left', 4: 'Talking on the phone - left', 5: 'Operating the radio', 6: 'Drinking', 7: 'Reaching behind', 8: 'Hair and makeup', 9: 'Talking to passenger'}

videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
result = True
counter = 0
while(result):
    time.sleep(5)
    name = "C:\\Users\\User\\Documents\\The Knowledge Society\\Innovate\\ai hackathon\\Using Artifical Intelligence to Detect Distracted Driving\\img\\test_" + str(counter) + ".jpg"
    ret,frame = videoCaptureObject.read()
    cv2.imwrite(name, frame)

    result = False

videoCaptureObject.release()
cv2.destroyAllWindows()

loaded_model = tf.keras.models.load_model("C:\\Users\\User\\Documents\\The Knowledge Society\\Innovate\\ai hackathon\\Using Artifical Intelligence to Detect Distracted Driving\\models\\model.h5")
open_img = Image.open(name).convert("RGB").resize((100,100)) #reaching back
pred_img = np.array(open_img, dtype="float32")
pred_img = np.expand_dims(pred_img, axis=0)
pred_img /= 255.0
pred_img = tf.convert_to_tensor(pred_img)
prediction.append(labels[np.argmax(loaded_model.predict(pred_img))])
plt.imshow(open_img)
plt.title(prediction)
plt.show()