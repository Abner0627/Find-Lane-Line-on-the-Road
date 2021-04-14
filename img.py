import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#%% Load video
P = './data'
V = 'challenge.mp4'
sV = 'Lane_' + V
vc = cv2.VideoCapture(os.path.join(P, V))
fps = vc.get(cv2.CAP_PROP_FPS)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
video = []

#%% Video process
for idx in range(frame_count):   
    vc.set(1, idx)
    ret, frame = vc.read()
    if idx==25:
        img = frame
        break
    
img = img[:,:,[2,1,0]]
plt.imshow(img)
plt.show()