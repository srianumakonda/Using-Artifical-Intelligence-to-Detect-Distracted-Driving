# Using-Artifical-Intelligence-to-Detect-Distracted-Driving

This project was made for TKS AI Hackathon in October 2020. Our goal was to identify a problem that can be solved using Artifical Intelligence. The problem that our team has 
come up with is the problem of distrcated driving. More than 50 million people are caught up in car accidents every year. What if there was a way we could prevent it?
Self driving cars aren't feasiable; they can't be implemented **right now**. We need to find a way to implement a method to detect distracted driving in our cars.

# The Dataset

Our team developed a Convolutional Neural Network that's trained on the [State Farm Distracted Driver Detection dataset on Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data).__
The 10 classes in this dataset include:

1. c0: safe driving
2. c1: texting - right
3. c2: talking on the phone - right
4. c3: texting - left
5. c4: talking on the phone - left
6. c5: operating the radio
7. c6: drinking
8. c7: reaching behind
9. c8: hair and makeup
10. c9: talking to passenger

Visual representation of all the classes (compressed in a gif):

![alt text](https://storage.googleapis.com/kaggle-competitions/kaggle/5048/media/output_DEb8oT.gif)

There are 1.1 million images. Approximately 90% of the images go to training (1 million images) with the other 10% (100,000 images) going to testing. THe 1 million images are then
divided into 10 classes leaving each class with ~100,000 images. The images are given in a jpg file. The dataset also includes 2 csv files which aren't used as they're not required for this specific project.  

#Data Preperation and Cleaning

Due to the fact that our dataset was a Kaggle dataset, it was very clean and easy to use. I used Tensorflow's ImageDataGenerator library to help upload the dataset and prepare it for training. In the jupyter notebook file, you will notice that I decided to play around with Data Augmentation and used that as my validation set. I wanted to see how the model would perform on augmented data which is why I wanted to test it. 
The way the data was collected was that a webcam was placed on the handle on the passenger's side. It would then take a picture of the driving performing certain actions and use that as the dataset. Having augmented data could help when the webcam isn't installed perfectly in a car and has a bit of angle to it.
I resized the images from a 640px by 480px to 100x by 100px. This will allow the model to process the image resulting in faster times while having enough detail to make a well enough prediction. 

# The Neural Network

As mentioned above, we use a Convolutional Neural Network to detect distracted driving in the dataset. Our neural network has 5 layers (1 input, 1 output, 3 hidden layers) with each hidden layer following this process:

- Convolution
- ReLU
- Maxpooling
- ReLU
- Dropout

This format is used for 3 layers. The kernal size for the convolution is kept the same (3x3, no padding) along with maxpooling (2x2, no padding). Dropout is kept at the same value for all 3 layers (0.1). 

Model summary:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 98, 98, 128)       3584      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 128)       0         
_________________________________________________________________
dropout (Dropout)            (None, 49, 49, 128)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 64)        73792     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 23, 23, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 23, 23, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 21, 32)        18464     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 10, 32)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 10, 10, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 3200)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              3277824   
_________________________________________________________________
dense_1 (Dense)              (None, 256)               262400    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 3,638,634
Trainable params: 3,638,634
Non-trainable params: 0
_________________________________________________________________
```

# Model Training

undistracted.world
