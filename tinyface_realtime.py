'''
captures video from webcam
runs the tinyface model on the sample video --> displays the tracked faces video in realtime 
writes the tracked faces video as 'video_tracked_tinyface_realtime.mp4'
'''

#import libraries
import torch
import numpy as np
import cv2
from detect import main_fn

#determine if an nvidia GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

checkpoint = 'weights/checkpoint_20.pth'
print('Checkpoint used for weights: ', checkpoint)

#get a sample video from webcam	
vid_capture = cv2.VideoCapture(0)

#frames = []
frames_tracked = []
while(vid_capture.isOpened()):
    #vid_capture.read() methods returns a tuple, first element is a bool 
    #and the second is frame	
    ret, frame = vid_capture.read()
    
    if ret == True:
        #convert frame to rgb format (from bgr of cv)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #run video through tinyface model to detect faces
        boxes = main_fn(image = rgb_frame, checkpoint = checkpoint , prob_thresh = 0.6, nms_thresh = 0.2)
    
        #draw the boundary box for the detected faces
        frame_draw = frame.copy()
        if boxes is not None: #incase no face is detected in a frame
            for box in boxes:
                box = box.astype(int)
                cv2.rectangle(frame_draw,(box[0],box[1]),(box[2],box[3]),(255, 0, 0), thickness = 4)

        #add to frame list
        frames_tracked.append(cv2.resize(frame_draw, (640,360), cv2.INTER_LINEAR))
        #frames.append(cv2.resize(frame, (640,360), cv2.INTER_LINEAR))
        
        #display the tracked video
        cv2.imshow('The Tracked Video',frame_draw)
        # 20 is in milliseconds
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

#release the video capture object
vid_capture.release()
cv2.destroyAllWindows()

#to write the tracked video
video_tracked = cv2.VideoWriter('video_tracked_tinyface_realtime.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 2, (640, 360))
for frame in frames_tracked:
    video_tracked.write(frame)
video_tracked.release()


