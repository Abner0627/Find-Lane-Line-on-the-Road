import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.linear_model import LinearRegression

#%% Load video
P = './data'
V = 'solidWhiteRight.mp4'
sP = './output'
sV = 'Lane_' + V
vc = cv2.VideoCapture(os.path.join(P, V))
fps = vc.get(cv2.CAP_PROP_FPS)
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
video = []

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


def linear_reg(x, y, res):
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y[:, np.newaxis])
    xfit = np.linspace(np.min(x), np.max(x), res)[:, np.newaxis]
    yfit = model.predict(xfit)
    return xfit, yfit

def _line(lines, res=2500):
    color = [255, 0, 0]
    thickness = 5
    x1 = lines[:, :, 0]
    y1 = lines[:, :, 1]
    x2 = lines[:, :, 2]
    y2 = lines[:, :, 3]
    lines_new = np.zeros((res*2, 1, 4))
    Lx1, Ly1, Rx1, Ry1 = lr_lines(x1, y1)
    Lx2, Ly2, Rx2, Ry2 = lr_lines(x2, y2)
    Lfitx1, Lfity1 = linear_reg(Lx1, Ly1, res)
    Lfitx2, Lfity2 = linear_reg(Lx2, Ly2, res)
    Rfitx1, Rfity1 = linear_reg(Rx1, Ry1, res)
    Rfitx2, Rfity2 = linear_reg(Rx2, Ry2, res)

    nx1 = np.concatenate((Lfitx1, Rfitx1)).astype(int)
    nx2 = np.concatenate((Lfitx2, Rfitx2)).astype(int)
    ny1 = np.concatenate((Lfity1, Rfity1)).astype(int)
    ny2 = np.concatenate((Lfity2, Rfity2)).astype(int)
    
    lines_new = np.concatenate((nx1, ny1, nx2, ny2), axis=1)[:, np.newaxis, :]

    return lines_new
    
#%% Video process
for idx in range(frame_count):   
    vc.set(1, idx)
    ret, frame = vc.read()
    if frame is not None:
        print('=====Frame >> ' + str(idx) + '/' + str(frame_count) + '=====', "\r" , end=' ')
        # print('=====Frame >> ' + str(idx) + '/' + str(frame_count) + '=====')
        img = frame
        
#%% Edge
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # kernel_size = 3
        # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
        # low_threshold = 1
        # high_threshold = 10
        # binary_warped = cv2.Canny(blur_gray, low_threshold, high_threshold)

        gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= 25) & (scaled_sobel <= 255)] = 1
        white_binary = np.zeros_like(gray_img)
        white_binary[(gray_img > 180) & (gray_img <= 255)] = 1
        binary_warped = cv2.bitwise_or(sx_binary, white_binary)

#%% Mask
        mask = np.zeros_like(binary_warped)   
        vertices = np.array(
                    [[153, 540],  # Bottom left
                    [430, 337],  # Top left
                    [536, 337],  # Top right
                    [872, 540]]) # Bottom right

        if len(binary_warped.shape) > 2:
            channel_count = binary_warped.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            

        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        masked_image = cv2.bitwise_and(binary_warped, mask)



#%% Line
        rho = 1
        theta = np.pi/180
        threshold = 1
        min_line_len = 10
        max_line_gap = 1

        lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
        lines_new = _line(lines)
        draw_lines(line_img, lines_new)


#%% Output
        result = cv2.addWeighted(line_img, 1, img, 1, 0)      
        height, width, layers = result.shape
        size = (width, height)
        video.append(result)
vc.release()

#%% Save video
out = cv2.VideoWriter(os.path.join(sP, sV), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(video)):
    out.write(video[i])
out.release()
      