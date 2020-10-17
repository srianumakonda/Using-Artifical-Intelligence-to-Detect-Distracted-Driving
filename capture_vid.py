import cv2
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# est = datetime.now(timezone('EST'))
# current_time = est.strftime("%y-%m-%d %H:%M")

videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
counter = 0

prediction = []
labels = {0: 'Safe driving', 1: 'Texting - right', 2: 'Talking on the phone - right', 3: 'Texting - left', 4: 'Talking on the phone - left', 5: 'Operating the radio', 6: 'Drinking', 7: 'Reaching behind', 8: 'Hair and makeup', 9: 'Talking to passenger'}
truee=True

while(truee):
    ret,frame = videoCaptureObject.read()
    name = "C:\\Users\\User\\Documents\\The Knowledge Society\\Innovate\\ai hackathon\\Using Artifical Intelligence to Detect Distracted Driving\\img\\" + str(counter) + ".jpg"
    cv2.imwrite(name ,frame)

    loaded_model = tf.keras.models.load_model("C:\\Users\\User\\Documents\\The Knowledge Society\\Innovate\\ai hackathon\\Using Artifical Intelligence to Detect Distracted Driving\\models\\model.h5")
    open_img = Image.open(name).convert("RGB").resize((100,100)) #reaching back
    pred_img = np.array(open_img, dtype="float32")
    pred_img = np.expand_dims(pred_img, axis=0)
    pred_img /= 255.0
    pred_img = tf.convert_to_tensor(pred_img)

    prediction.append(labels[np.argmax(loaded_model.predict(pred_img))])
    plt.imshow(open_img)
    plt.title(prediction)

    time.sleep(5)
    counter+=1
    truee=False
videoCaptureObject.release()
cv2.destroyAllWindows()

plt.show()