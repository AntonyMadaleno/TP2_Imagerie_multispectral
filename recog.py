# Import necessary libraries
import numpy as np
import cv2

from bilinear import *
from mosaicage import *

# Define a function for demosaicing using a specific method
def form_recognition(cfa):
    
    # Create a Bayer mosaic pattern
    calque = bayer_mosaicage(np.ones(cfa.shape, dtype=np.uint8))
    R = cfa[:, :, 0]
    G = cfa[:, :, 1]
    B = cfa[:, :, 2]

    res = np.zeros(cfa.shape, dtype=np.uint8)

    # Apply demosaicing for the green channel
    kernel_g = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    tmp = cv2.filter2D(G, 0, kernel_g, borderType=cv2.BORDER_REFLECT)

    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):

            if G[i, j] == 0:
                # Create a contour matrix to check neighboring pixels
                contour = np.zeros((3, 3), dtype=bool)
                m = tmp[i, j]
                gs = [G[i, j - 1], G[i + 1, j], G[i, j + 1], G[i - 1, j]]

                contour[1, 0] = gs[0] >= m
                contour[2, 1] = gs[1] >= m
                contour[1, 2] = gs[2] >= m
                contour[0, 1] = gs[3] >= m

                t = np.sum(contour)
                
                if t == 3:
                    res[i, j, 1] = st.median(np.array(gs).astype(np.double))
                else:
                    gs.sort()

                    # Create a 5x5 array
                    nc = np.full((5, 5), 1/8)
                    # Set specific elements to 0
                    nc[1:4, 1:4] = 0

                    # Create a cropped view of the G array
                    i_min = max(0, i - 2)
                    i_max = min(i + 3, G.shape[0] - 1)
                    j_min = max(0, j - 2)
                    j_max = min(j + 3, G.shape[1] - 1)
                    cp = G[i_min:i_max, j_min:j_max]
                   
                    if cp.shape[1] < 5:
                        if j > 1:
                            cp = np.c_[cp, cp[:, cp.shape[1] - 1]]
                        else:
                            cp = np.c_[cp, cp[:, 0]]

                    if cp.shape[0] < 5:
                        if i > 1:
                            cp = np.r_['0,2', cp, cp[cp.shape[0] - 1, :]]
                        else:
                            cp = np.r_['0,2', cp, cp[0, :]]

                    # Handle different cases (band and corner)
                    if contour[1, 0] == contour[1, 2]:
                        s = cv2.filter2D(cp, 0, nc)[2, 2]
                        res[i, j, 1] = max(min(gs[1], (2 * m - s)), gs[2])
                    else:
                        if contour[1, 0] == contour[0, 1]:
                            nc[1, 0] = 0
                            nc[0, 1] = 0
                            nc[3, 4] = 0
                            nc[4, 3] = 0
                        else:
                            nc[3, 0] = 0
                            nc[4, 1] = 0
                            nc[1, 4] = 0
                            nc[0, 3] = 0

                        nc = nc * 2
                        s = cv2.filter2D(cp, 0, nc)[2, 2]
                        res[i, j, 1] = max(min(gs[1], (2 * m - s)), gs[2])

    res[:, :, 1] = res[:, :, 1] + G
    bil = bilinear_loop(cfa)
    res[:, :, 0] = bil[:, :, 0]
    res[:, :, 2] = bil[:, :, 2]

    return res.astype(np.uint8)