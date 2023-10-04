# Import necessary libraries
import numpy as np
import cv2

# Define a function for bilinear interpolation using loops
def bilinear_loop(cfa):
    result = np.zeros(cfa.shape, dtype=np.uint8)
    R = cfa[:, :, 0]
    G = cfa[:, :, 1]
    B = cfa[:, :, 2]

    # Define a helper function for bilinear interpolation
    def helper(C, i, j, n):
        shape = C.shape
        sum = 0

        # Loop through neighboring pixels for interpolation
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

                if C[ci, cj] >= 0:
                    sum += C[ci, cj]
        
        # Calculate the average value for interpolation
        sum = sum / n
        return sum

    # Loop through each pixel in the image
    for i in range(0, cfa.shape[0]):
        for j in range(0, cfa.shape[1]):
            f = [
                1 - ((i + j) % 2 + j % 2 * i % 2),
                (i + j) % 2,
                j % 2 * i % 2,
            ]

            if f[0] == 0:
                if f[2] == 1:
                    result[i, j, 0] = helper(R, i, j, 4)
                else:
                    result[i, j, 0] = helper(R, i, j, 2)

            if f[1] == 0:
                result[i, j, 1] = helper(G, i, j, 4)

            if f[2] == 0:
                if f[0] == 1:
                    result[i, j, 2] = helper(B, i, j, 4)
                else:
                    result[i, j, 2] = helper(B, i, j, 2)

    return (result + cfa).astype(np.uint8)

# Define a function for bilinear interpolation using convolution
def bilinear_convolution(cfa):
    # Define convolution kernels for bilinear interpolation
    kernel_rb = np.array([[0.25, 0.5, 0.25],
                         [0.5, 1, 0.5],
                         [0.25, 0.5, 0.25]], dtype=np.float32)

    kernel_g = np.array([[0, 0.25, 0],
                         [0.25, 1, 0.25],
                         [0, 0.25, 0]], dtype=np.float32)

    # Apply convolution to each channel
    result_r = cv2.filter2D(cfa[:, :, 0], 0, kernel_rb, borderType=cv2.BORDER_REFLECT)
    result_g = cv2.filter2D(cfa[:, :, 1], 0, kernel_g, borderType=cv2.BORDER_REFLECT)
    result_b = cv2.filter2D(cfa[:, :, 2], 0, kernel_rb, borderType=cv2.BORDER_REFLECT)

    # Combine the channels to get the final result
    result = cv2.merge([result_r, result_g, result_b])

    return result.astype(np.uint8)