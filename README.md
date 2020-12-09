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

# The Neural Network

As mentioned above, we use a Convolutional Neural Network to detect distracted driving in the dataset. Our neural network has 5 layers (1 input, 1 output, 3 hidden layers) with each hidden layer following this process:

- Convolution
- ReLU
- Maxpooling
- ReLU
- Dropout

This format is used for 3 layers
undistracted.world
