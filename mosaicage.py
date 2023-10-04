# Import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics as st
import os

# Define a function for Bayer Mosaicing
def bayer_mosaicage(img):
    # Create an empty image of the same shape as the input image
    res = np.zeros(img.shape, dtype=np.uint8)

    # Loop through each pixel in the image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # Define a Bayer filter pattern
            filtre = [
                1 - ((i + j) % 2 + j % 2 * i % 2),
                (i + j) % 2,
                j % 2 * i % 2,
            ]

            # Apply the filter to the input image and store the result
            res[i, j, :] = img[i, j, :] * np.array(filtre)
    
    return res

# Define a function to calculate Mean Squared Error (MSE) between two images
def MSE(i1, i2):
    return np.mean((i1 - i2)**2)

# Define a function to calculate local MSE
def MSE_local(i1, i2, bsize=16):
    x_block_count = i1.shape[1] // bsize
    y_block_count = i1.shape[0] // bsize

    mse_map = np.zeros((y_block_count, x_block_count))
    
    for y in range(0, y_block_count):
        for x in range(0, x_block_count):
            bs_x = x * bsize
            bs_y = y * bsize
            be_x = bs_x + bsize
            be_y = bs_y + bsize

            mse_map[y, x] = MSE(i1[bs_y:be_y, bs_x:be_x], i2[bs_y:be_y, bs_x:be_x])
    
    return mse_map
