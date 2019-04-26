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
Multi Task Cascaded Convolutional Networks performs face detection along with detection landmarks (eyes, nose & mouth endpoints). They provide a very high accuracy in real time. MTCNN leverages cascaded architectures with three stages of carefully designed deep convolutional networks to predict the same.
Refer to [README.md inside mtcnn_src dir](./mtcnn_src/README.md) to understand its model architecture.

## Overview of the Model

## Training on your own dataset

## References
- MTCNN paper link: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878v1.pdf)
- MTCNN Web Link: https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
