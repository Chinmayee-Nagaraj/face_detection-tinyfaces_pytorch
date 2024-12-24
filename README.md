# Real-Time Face Detection using Tiny Faces in PyTorch

This repository implements a real-time face detection pipeline by using the TinyFaces PyTorch model from [Tiny Faces in PyTorch](https://github.com/varunagrawal/tiny-faces-pytorch). 

The original [facenet-pytorch face tracking pipeline](https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb) is modified and MTCNN is replaced with the TinyFaces PyTorch model. 

Key changes include using OpenCV (cv2) for video capture and substituting ResNet101 with ResNet50 as the encoder for training the TinyFaces model.


## Model Training 
* The TinyFaces PyTorch model from [Tiny Faces in PyTorch](https://github.com/varunagrawal/tiny-faces-pytorch) was trained using the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) data.
* The encoder architecture with  ResNet101 was replaced with ResNet50.
* After training for 20 epoches, the checkpoint_20.pth is used for the face detection.
  
* The training curve for classification loss and regression loss are as below : 
![Screenshot from 2024-12-23 16-44-38](https://github.com/user-attachments/assets/4c229a92-9a67-4aaf-a2d3-87f73f5644e1)
![Screenshot from 2024-12-23 16-44-55](https://github.com/user-attachments/assets/83339a94-2f1c-4792-874c-0d804c0810b5)


## Model Evaluation
* The evaluation is done and the output files are generated as per the WIDERFace specification .
* The mAP results for the trained model, on the WIDER Face dataset was obtained using [WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation.git) as below:
~~~
==================== Results ====================
Easy   Val AP: 0.6962241504550841
Medium Val AP: 0.7234407604696959
Hard   Val AP: 0.5412494164148824
=================================================
~~~


## Results
### Face Detection on a test image, using the trained model:
![tracked_image](https://github.com/user-attachments/assets/203dac44-9894-4b61-98d4-173e78225bd3)


# Getting Started
* Clone this repository.
* Download the checkpoint obtained after training the tinyface model for 20 epoches, from [here](https://drive.google.com/drive/folders/1Z-NWrzt1nRNWnZzCdLl9VvpcacDYYTq2?usp=sharing) into the weights folder; such that:
~~~
$ ls weights/
checkpoint_20.pth
~~~
## Run the model
* At the repo root type the below to get real-time face detection, through webcam.
~~~
$ python tinyface_realtime.py
~~~
* Press 'q' to stop the running the model. The video is stored as 'video_tracked_tinyface_realtime.mp4' in the repo root.
  

