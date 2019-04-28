# face_verification_demo
This is a simple example for face verification using [facenet : A unified Embedding for face recognition &amp; clustering](http://arxiv.org/abs/1503.03832) implemented by davidsandberg.

## Requirements
```
Linux (Tested on Windows 10)
Python
Python Packages
 - numpy
 - opencv-python
 - TensorFlow
 - scipy
 - sklearn
 - pickle
```
you can install the python package using `pip install <package_name>`


## Inspiration
I came across FaceNet network, which performs the task of face recognition, verification & clustering. So I decided to try my
hands on this fancy stuff as well.

## Model Download (pretrained)
- Download facenet model from [here](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) and copy to `model` directory
- Download the MTCNN model from [here] and place it under `mtcnn_model` directory.
After Download and extracting your directories should look something like this
```
face_verification_demo
├─ mtcnn_model
|   ├─ all_in_one
│   |   ├─ checkpoint.txt
│   |   ├─ mtcnn-3000000.data-00000-of-00001
│   |   ├─ mtcnn-3000000.index
│   |   └─ mtcnn-3000000.meta
├─ model
|   ├─ 20180402-114759
│   |   ├─ 20180402-114759.pb
│   |   ├─ model-20180402-114759.ckpt-275.data-00000-of-00001
│   |   ├─ model-20180402-114759.ckpt-275.index
│   |   ├─ model-20180402-114759.meta
└─ ...
```

## Preprocessing using MTCNN Face Alignment
Multi Task Cascaded Convolutional Networks performs face detection along with detection landmarks (eyes, nose & mouth endpoints). 
They provide a very high accuracy in real time. MTCNN leverages cascaded architectures 
with three stages of carefully designed deep convolutional networks to predict the same.
Refer to [README.md inside mtcnn_src dir](./mtcnn_src/README.md) to understand its model architecture.

Therefore MTCNN is used to crop faces from the images, which in turn is fed as input to our FaceNet Network.

## Overview of the FaceNet Model
FaceNet is a neural network that learns a mapping from images to a compact Eucledian space where distance correspond to measure of 
face similarity. Therefore similar face images will have lesser distance between them and vice versa. Once we have the feature vector/
Eucledian embedding per image using the deep convolutional network (FaceNet), 
tasks such as face recognition, clustering & verification can easily be implemented.
<p align="center">
	<img src="./mtcnn_src/images/facenet_arch.png">
</p>

The advantage of using the 128D embedding is that you don't need to retrain you model to recoginize new faces.
All you need is a single image for an individual to get that embedding once passed throught the network.
That embedding can be stored and used as a reference for any future queries. This makes FaceNet even more powerful.

### Triplet Loss
FaceNet uses a distinct loss method called Triplet Loss to calculate loss. 
Triplet Loss minimises the distance between an anchor and a positive, 
images that contain same identity, and maximises the distance between the anchor and a negative, images that contain different identities.
<p align="center">
	<img src="./mtcnn_src/images/triplet_loss.png">
</p>

Conceptually it means, Faces of same identity should appear closer to each other than faces of another identity.
## Implementation
### 1. Database
Here is what the `images` directory looks like:
<p align="center">
	<img src="./mtcnn_src/images/database.png">
</p>

Database has several images of my friends, for which we will generate the 128D embedding and dump it in a pickle file.


Simply add your own set of images to `./images` directory for performing facial recognition on your own set of images.


In my implementation, I have used tensorflow. Additionally I am using utility file `facenet.py` to abstract all interactions with 
FaceNet network. The file is responsible for loading the network and have api's to perform inference.


## References
- MTCNN paper link: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878v1.pdf)
- MTCNN Web Link: https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
- https://github.com/wangbm/MTCNN-Tensorflow
- https://github.com/davidsandberg/facenet
- https://github.com/amenglong/face_recognition_cnn
- https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8
- https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
- https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c