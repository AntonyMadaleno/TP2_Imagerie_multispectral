import cv2
import numpy as np

import numpy as np

import numpy as np

def hue_constance(img):
    result = np.zeros(img.shape, dtype=np.uint8)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    shape = R.shape
    #Loop through all pixels without green
    for i in range(0, img.shape[0]):
        for j in range(i%2, img.shape[1],2):
            sg = 0
            #Loop through the neighbouring pixels
            for x in range(-1, 2):
                for y in range(-1, 2):
                    ci = x + i
                    cj = y + j

                    # Handle boundary conditions
                    if ci < 0:
                        ci = 1
                    elif ci >= shape[0]:
                        ci = shape[0] - 2
                    if cj < 0:
                        cj = 1
                    elif cj >= shape[1]:
                        cj = shape[1] - 2
                    if G[ci, cj] >= 0:
                        sg += G[ci,cj]
            result[i,j,1] = sg/4
    # Loop through each pixel in the image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            f = [
                int((i%2 + j%2)==0),
                (i + j) % 2,
                j % 2 * i % 2,
            ]
            #Initialize the sum for each color
            sr = 0
            sb = 0
            #Loop through the neighbouring pixels
            for x in range(-1, 2):
                for y in range(-1, 2):
                    ci = x + i
                    cj = y + j

                    # Handle boundary conditions
                    if ci < 0:
                        ci = 1
                    elif ci >= shape[0]:
                        ci = shape[0] - 2
                    if cj < 0:
                        cj = 1
                    elif cj >= shape[1]:
                        cj = shape[1] - 2
                    #Calculation of the sum of red/blue pixels weighted by the value of green pixel
                    if R[ci, cj] >= 0:
                        sr += R[ci, cj]/max(1,G[ci, cj],result[ci, cj,1])
                    if B[ci, cj] >= 0:
                        sb += B[ci, cj]/max(1,G[ci, cj],result[ci, cj,1])
            #Calculation of the new value of red/blue pixels

            if f[1]==0:
                if f[0]==0:
                    result[i,j,0] = min(255,max(1,result[i,j,1])/4 * sr)
                else:
                    result[i,j,2] = min(255,max(1,result[i,j,1])/4 * sb)
            else:
                result[i,j,0] = min(255,max(1,G[i,j])/2 * sr)
                result[i,j,2] = min(255,max(1,G[i,j])/2 * sb)
                
    return (result + img).astype(np.uint8)
