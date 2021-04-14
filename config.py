import numpy as np
WR_vertices = np.array(
    [[0, 540],  # Bottom left
    [425, 330],  # Top left
    [535, 330],  # Top right
    [960, 540]]) # Bottom right

YL_vertices = np.array(
    [[0, 540],  # Bottom left
    [430, 330],  # Top left
    [535, 330],  # Top right
    [960, 540]]) # Bottom right

CH_vertices = np.array(
    [[280, 666],  # Bottom left
    [595, 460],  # Top left
    [735, 460],  # Top right
    [1080, 666]]) # Bottom right

N1_vertices = np.array(
    [[0, 990],  # Bottom left
    [805, 700],  # Top left
    [1040, 700],  # Top right
    [1920, 990]]) # Bottom right    


rho = 1
theta = np.pi/180
threshold = 1
min_line_len = 10
max_line_gap = 1        