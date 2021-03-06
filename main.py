#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.linear_model import LinearRegression

import argparse
import config

#%% Arg
parser = argparse.ArgumentParser()
parser.add_argument('-V','--video',
                   default='solidWhiteRight',
                   help='input video file name')

args = parser.parse_args()

#%% Load video
# data path
P = './data'
V = args.video + '.mp4'
# save path
sP = './output'
sV = 'Lane_' + V
# capture video and calculate its fps
vc = cv2.VideoCapture(os.path.join(P, V))
fps = vc.get(cv2.CAP_PROP_FPS)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
video = []

#%% Parameters
rho = config.rho
theta = config.theta
threshold = config.threshold
min_line_len = config.min_line_len
max_line_gap = config.max_line_gap

if args.video == 'solidWhiteRight':
    vertices = config.WR_vertices
elif args.video == 'challenge':
    vertices = config.CH_vertices
elif args.video == 'solidYellowLeft':
    vertices = config.YL_vertices
elif args.video == 'tw_NH1':
    vertices = config.N1_vertices                                          
else:
    print('Wrong file name')
    exit()

#%% Functions
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def lr_lines(x, y):
    Lx, Ly, Rx, Ry = [], [], [], []
    Llimt, Rlimt = vertices[1, 0], vertices[2, 0]

    for xi, yi in zip(x[:,0], y[:,0]):
        if xi < ((Llimt + Rlimt)//2):
            Lx.append(xi)
            Ly.append(yi)
        else:
            Rx.append(xi)
            Ry.append(yi)
    return np.array(Lx), np.array(Ly), np.array(Rx), np.array(Ry)


def linear_reg(x, y, res, Llimt, Rlimt):
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y[:, np.newaxis])
    xfit = np.linspace(Llimt, Rlimt, res)[:, np.newaxis]
    yfit = model.predict(xfit)
    return xfit, yfit

def _line(lines, vertices, lines_new, res=2500):
    x1 = lines[:, :, 0]
    y1 = lines[:, :, 1]
    x2 = lines[:, :, 2]
    y2 = lines[:, :, 3]
    lines_out = np.zeros((res*2, 1, 4))
    Lx1, Ly1, Rx1, Ry1 = lr_lines(x1, y1)
    Lx2, Ly2, Rx2, Ry2 = lr_lines(x2, y2)
    # find the minimum number in lane line pixels
    min_Lpt = np.min([Lx1.size, Ly1.size, Lx2.size, Ly2.size])
    min_Rpt = np.min([Rx1.size, Ry1.size, Rx2.size, Ry2.size])
    # if its amount was equal to zero, uses fitting lines of previous frame
    if min_Lpt==0 or min_Rpt==0:
        lines_out = lines_new
    
    elif args.video == 'challenge':
        if min_Lpt<150 or min_Rpt<40:
            lines_out = lines_new
    else:
        Lfitx1, Lfity1 = linear_reg(Lx1, Ly1, res, vertices[0, 0], vertices[1, 0])
        Lfitx2, Lfity2 = linear_reg(Lx2, Ly2, res, vertices[0, 0], vertices[1, 0])
        Rfitx1, Rfity1 = linear_reg(Rx1, Ry1, res, vertices[3, 0], vertices[2, 0])
        Rfitx2, Rfity2 = linear_reg(Rx2, Ry2, res, vertices[3, 0], vertices[2, 0])

        nx1 = np.concatenate((Lfitx1, Rfitx1))
        nx2 = np.concatenate((Lfitx2, Rfitx2))
        ny1 = np.concatenate((Lfity1, Rfity1))
        ny2 = np.concatenate((Lfity2, Rfity2))
        
        nx = ((nx1 + nx2)/2).astype(int)
        ny = ((ny1 + ny2)/2).astype(int)

        lines_out = np.concatenate((nx, ny, nx, ny), axis=1)[:, np.newaxis, :]

    return lines_out
    
#%% Video process
lines_new = 0
for idx in range(frame_count):   
    vc.set(1, idx)
    ret, frame = vc.read()
    if frame is not None:
        print('=====Frame >> ' + str(idx) + '/' + str(frame_count) + '=====', "\r" , end=' ')
        # print('=====Frame >> ' + str(idx) + '/' + str(frame_count) + '=====')
        img = frame
        
#%% Edge
        # transform into gray scale img
        gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # sobel filter
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        if args.video == 'solidWhiteRight':
            sx_binary = np.zeros_like(scaled_sobel)
            sx_binary[(scaled_sobel >= 25) & (scaled_sobel <= 255)] = 1
            white_binary = np.zeros_like(gray_img)
            white_binary[(gray_img > 180) & (gray_img <= 255)] = 1
            binary_warped = cv2.bitwise_or(sx_binary, white_binary)

        elif args.video == 'solidYellowLeft':
            sx_binary = np.zeros_like(scaled_sobel)
            sx_binary[(scaled_sobel >= 15) & (scaled_sobel <= 255)] = 1
            white_binary = np.zeros_like(gray_img)
            white_binary[(gray_img > 200) & (gray_img <= 255)] = 1
            binary_warped = cv2.bitwise_or(sx_binary, white_binary)

        elif args.video == 'challenge':
            sx_binary = np.zeros_like(scaled_sobel)
            sx_binary[(scaled_sobel >= 15) & (scaled_sobel <= 255)] = 1
            white_binary = np.zeros_like(gray_img)
            white_binary[(gray_img > 180) & (gray_img <= 255)] = 1
            binary_warped = cv2.bitwise_or(sx_binary, white_binary)

        elif args.video == 'tw_NH1':
            sx_binary = np.zeros_like(scaled_sobel)
            sx_binary[(scaled_sobel >= 25) & (scaled_sobel <= 255)] = 1
            white_binary = np.zeros_like(gray_img)
            white_binary[(gray_img > 180) & (gray_img <= 255)] = 1
            binary_warped = cv2.bitwise_or(sx_binary, white_binary)

        else:
            # set binary threshold of sobel
            sx_binary = np.zeros_like(scaled_sobel)
            sx_binary[(scaled_sobel >= 25) & (scaled_sobel <= 255)] = 1
            # set binary threshold of gray scale img
            white_binary = np.zeros_like(gray_img)
            white_binary[(gray_img > 150) & (gray_img <= 255)] = 1
            # combine them with 'OR'
            binary_warped = cv2.bitwise_or(sx_binary, white_binary)                        

#%% Mask
        # initailize mask
        mask = np.zeros_like(binary_warped)   
        if len(binary_warped.shape) > 2:
            channel_count = binary_warped.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        # fill mask region
        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        masked_image = cv2.bitwise_and(binary_warped, mask)

#%% Line
        # use Hough transform to find the lane lines
        lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]),
                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
        # divide the left part or right part of lane line, 
        # then use linear regression to find their fitting lines
        lines_new = _line(lines, vertices, lines_new)
        # draw the fiiting lines
        draw_lines(line_img, lines_new)


#%% Output
        line_img = line_img[:,:,[2,1,0]]
        # map overlay
        result = cv2.addWeighted(img, 1, line_img, 1, 0)      
        height, width, layers = result.shape
        # set the size of output video
        size = (width, height)
        video.append(result)

#%% Save video
vc.release()
out = cv2.VideoWriter(os.path.join(sP, sV), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(video)):
    out.write(video[i])
out.release()
      