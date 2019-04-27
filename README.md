# face_verification_demo
This is a simple example for face verification using [facenet : A unified Embedding for face recognition &amp; clustering](http://arxiv.org/abs/1503.03832) implemented by davidsandberg.

## Requirements
```
Linux (Tested on CentOS 7)
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

## Inspiration

## MTCNN Face Alignment
Multi Task Cascaded Convolutional Networks performs face detection along with detection landmarks (eyes, nose & mouth endpoints). 
They provide a very high accuracy in real time. MTCNN leverages cascaded architectures 
with three stages of carefully designed deep convolutional networks to predict the same.
Refer to [README.md inside mtcnn_src dir](./mtcnn_src/README.md) to understand its model architecture.

## Overview of the FaceNet Model
FaceNet is a neural network that learns a mapping from images to a compact Eucledian space where distance correspond to measure of 
face similarity. Therefore similar face images will have lesser distance between them and vice versa.
### Triplet Loss
FaceNet uses a distinct loss method called Triplet Loss to calculate loss. 
Triplet Loss minimises the distance between an anchor and a positive, 
images that contain same identity, and maximises the distance between the anchor and a negative, images that contain different identities.
FaceNet is a Siamese Network. To understand more about it, refer [here]()

### Implementation
In my implementation, I have used tensorflow. Additionally I am using utility file `facenet.py` to abstract all interactions with 
FaceNet network. The file is responsible for loading the network and have api's to perform inference.

`faces` is a list of dictioniary. Its output structure would look like:
```
faces [{'rect': [357, 129, 479, 315], 'embedding': array([[-8.19484070e-02, -2.21129805e-02, -1.30576044e-01,
		 6.40663598e-03, -1.72499865e-02, -2.30523646e-02,
         4.76869754e-02,  7.95004666e-02,  5.91421779e-03,
         1.85606722e-02,  6.09219307e-04,  2.72919453e-04,
		 .
		 .
		 .
		 -8.93132612e-02, -1.93766430e-02]], dtype=float32), 'name': 'Akansha'}]
```

## Training on your own dataset

## References
- MTCNN paper link: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878v1.pdf)
- MTCNN Web Link: https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
