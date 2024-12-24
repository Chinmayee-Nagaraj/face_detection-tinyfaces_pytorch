# Real-Time Face Detection using Tiny Faces in PyTorch

This project implements a real-time face detection pipeline by using the TinyFaces PyTorch model from [Tiny Faces in PyTorch](https://github.com/varunagrawal/tiny-faces-pytorch). The original [facenet-pytorch face tracking pipeline](https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb) is modified and MTCNN is replaced with the TinyFaces PyTorch model. Key changes include using OpenCV (cv2) for video capture and substituting ResNet101 with ResNet50 as the encoder for training the TinyFaces model.

## Getting Started
* Clone this repository.
* Download the checkpoint obtained after training the tinyface model for 20 epoches, from [here](https://drive.google.com/drive/folders/1Z-NWrzt1nRNWnZzCdLl9VvpcacDYYTq2?usp=sharing) into the weights folder; such that:
~~~
$ ls weights/
checkpoint_20.pth
~~~
* 


