import numpy as np

import numpy as np

def preservation_contour(img):
    # Create a result array of double precision
    result = np.zeros(img.shape, dtype=np.double)

    # Convert R, G, and B channels to double precision
    R = img[:, :, 0].astype(np.double)
    G = img[:, :, 1].astype(np.double)
    B = img[:, :, 2].astype(np.double)

    shape = R.shape

    # Loop through all pixels without green
    for i in range(0, img.shape[0]):
        for j in range(i % 2, img.shape[1], 2):
            g1 = 0
            g2 = 0

            if j % 2 * i % 2 == 1:
                # Calculate indices for green and red/blue pixel neighbors
                v1 = i - 1
                v2 = i + 1
                r1 = i - 2
                r2 = i + 2

                # Handle boundary conditions
                v1 = abs(v1) if v1 < 0 else v1
                v2 = 2 * (shape[0] - 1) - v2 if v2 >= shape[0] else v2
                r1 = abs(r1) if r1 < 0 else r1
                r2 = 2 * (shape[0] - 1) - r2 if r2 >= shape[0] else r2

                # Calculate g1 and g2
                g1 = abs(G[v1, j] - G[v2, j]) + abs(B[i, j] - B[r1, j] + B[i, j] - B[r2, j])

                # Calculate indices for green and red/blue pixel neighbors in the horizontal direction
                v3 = j - 1
                v4 = j + 1
                r3 = j - 2
                r4 = j + 2

                # Handle boundary conditions
                v3 = abs(v3) if v3 < 0 else v3
                v4 = 2 * (shape[1] - 1) - v4 if v4 >= shape[1] else v4
                r3 = abs(r3) if r3 < 0 else r3
                r4 = 2 * (shape[1] - 1) - r4 if r4 >= shape[1] else r4

                # Calculate g2
                g2 = abs(G[i, v3] - G[i, v4]) + abs(B[i, j] - B[i, r3] + B[i, j] - B[i, r4])

                # Update the result based on g1 and g2
                if g1 > g2:
                    result[i, j, 1] = max(0, min(255, (G[i, v3] + G[i, v4]) / 2 + (B[i, j] - B[i, r3] + B[i, j] - B[i, r4]) / 4))
                elif g2 > g1:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j]) / 2 + (B[i, j] - B[r1, j] + B[i, j] - B[r2, j]) / 4))
                else:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j] + G[i, v3] + G[i, v4]) / 4 + (B[i, j] - B[r1, j] + B[i, j] - B[r2, j] + B[i, j] - B[i, r3] + B[i, j] - B[i, r4]) / 8))
            else:
                # Calculate indices for green and red/blue pixel neighbors
                v1 = i - 1
                v2 = i + 1
                r1 = i - 2
                r2 = i + 2

                # Handle boundary conditions
                v1 = abs(v1) if v1 < 0 else v1
                v2 = 2 * (shape[0] - 1) - v2 if v2 >= shape[0] else v2
                r1 = abs(r1) if r1 < 0 else r1
                r2 = 2 * (shape[0] - 1) - r2 if r2 >= shape[0] else r2

                # Calculate g1 using green and red pixel neighbors
                g1 = abs(G[v1, j] - G[v2, j]) + abs(R[i, j] - R[r1, j] + R[i, j] - R[r2, j])

                # Calculate indices for green and red/blue pixel neighbors in the horizontal direction
                v3 = j - 1
                v4 = j + 1
                r3 = j - 2
                r4 = j + 2

                # Handle boundary conditions
                v3 = abs(v3) if v3 < 0 else v3
                v4 = 2 * (shape[1] - 1) - v4 if v4 >= shape[1] else v4
                r3 = abs(r3) if r3 < 0 else r3
                r4 = 2 * (shape[1] - 1) - r4 if r4 >= shape[1] else r4

                # Calculate g2 using green and red/blue pixel neighbors in the horizontal direction
                g2 = abs(G[i, v3] - G[i, v4]) + abs(R[i, j] - R[i, r3] + R[i, j] - R[i, r4])

                # Update the result based on g1 and g2
                if g1 > g2:
                    result[i, j, 1] = max(0, min(255, (G[i, v3] + G[i, v4]) / 2 + (R[i, j] - R[i, r3] + R[i, j] - R[i, r4]) / 4))
                elif g2 > g1:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j]) / 2 + (R[i, j] - R[r1, j] + R[i, j] - R[r2, j]) / 4))
                else:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j] + G[i, v3] + G[i, v4]) / 4 + (R[i, j] - R[r1, j] + R[i, j] - R[r2, j] + R[i, j] - R[i, r3] + R[i, j] - R[i, r4]) / 8))

    # Loop through each pixel in the image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            f = [
                int((i % 2 + j % 2) == 0),
                (i + j) % 2,
                j % 2 * i % 2,
            ]

            # Initialize the sum for each color
            sr = 0
            sb = 0

            # Loop through the neighboring pixels
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

                    # Calculation of the sum of red/blue pixels weighted by the value of green pixel
                    if R[ci, cj] >= 0:
                        sr += R[ci, cj] / max(1, G[ci, cj], result[ci, cj, 1])
                    if B[ci, cj] >= 0:
                        sb += B[ci, cj] / max(1, G[ci, cj], result[ci, cj, 1])

            # Calculation of the new value of red/blue pixels
            if f[1] == 0:
                if f[0] == 0:
                    result[i, j, 0] = min(255, max(1, result[i, j, 1]) / 4 * sr)
                else:
                    result[i, j, 2] = min(255, max(1, result[i, j, 1]) / 4 * sb)
            else:
                result[i, j, 0] = min(255, max(1, G[i, j]) / 2 * sr)
                result[i, j, 2] = min(255, max(1, G[i, j]) / 2 * sb)

    return (result + img).astype(np.uint8)

def preservation_contour2(img):
    # Create an empty result image with double precision
    result = np.zeros(img.shape, dtype=np.double)

    # Extract the Red, Green, and Blue channels and convert them to double precision
    R = img[:, :, 0].astype(np.double)
    G = img[:, :, 1].astype(np.double)
    B = img[:, :, 2].astype(np.double)
    
    # Get the shape of the image
    shape = R.shape

    # Loop through all pixels without green
    for i in range(0, img.shape[0]):
        for j in range(i % 2, img.shape[1], 2):
            g1 = 0
            g2 = 0
            
            # Check if the pixel is of type "odd"
            if j % 2 * i % 2 == 1:
                # Define neighboring pixel indices for green and blue channels
                v1 = max(0, i - 1)
                v2 = min(shape[0] - 1, i + 1)
                r1 = max(0, i - 2)
                r2 = min(shape[0] - 1, i + 2)

                # Calculate the absolute differences in the green channel
                g1 = abs(G[v1, j] - G[v2, j])
                
                # Define neighboring pixel indices for green and blue channels
                v3 = max(0, j - 1)
                v4 = min(shape[1] - 1, j + 1)
                r3 = max(0, j - 2)
                r4 = min(shape[1] - 1, j + 2)

                # Calculate the absolute differences in the green channel
                g2 = abs(G[i, v3] - G[i, v4])

                # Determine the condition for selecting g1 or g2
                if g1 > g2:
                    result[i, j, 1] = max(0, min(255, (G[i, v3] + G[i, v4]) / 2))
                elif g2 > g1:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j]) / 2))
                else:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j] + G[i, v3] + G[i, v4]) / 4))
            
            # If the pixel is not of type "odd"
            else:
                # Define neighboring pixel indices for green and blue channels
                v1 = max(0, i - 1)
                v2 = min(shape[0] - 1, i + 1)
                r1 = max(0, i - 2)
                r2 = min(shape[0] - 1, i + 2)

                # Calculate the absolute differences in the green channel
                g1 = abs(G[v1, j] - G[v2, j])

                # Define neighboring pixel indices for green and blue channels
                v3 = max(0, j - 1)
                v4 = min(shape[1] - 1, j + 1)
                r3 = max(0, j - 2)
                r4 = min(shape[1] - 1, j + 2)

                # Calculate the absolute differences in the green channel
                g2 = abs(G[i, v3] - G[i, v4])

                # Determine the condition for selecting g1 or g2
                if g1 > g2:
                    result[i, j, 1] = max(0, min(255, (G[i, v3] + G[i, v4]) / 2))
                elif g2 > g1:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j]) / 2))
                else:
                    result[i, j, 1] = max(0, min(255, (G[v1, j] + G[v2, j] + G[i, v3] + G[i, v4]) / 4))

    # Loop through each pixel in the image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            f = [
                int((i % 2 + j % 2) == 0),
                (i + j) % 2,
                j % 2 * i % 2,
            ]
            
            # Initialize the sum for the red and blue channels
            sr = 0
            sb = 0
            
            # Loop through the neighboring pixels
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
                    
                    # Calculation of the sum of red/blue pixels weighted by the value of the green pixel
                    if R[ci, cj] >= 0:
                        sr += R[ci, cj] / max(1, G[ci, cj], result[ci, cj, 1])
                    if B[ci, cj] >= 0:
                        sb += B[ci, cj] / max(1, G[ci, cj], result[ci, cj, 1])

            # Calculation of the new value of red/blue pixels based on the conditions
            if f[1] == 0:
                if f[0] == 0:
                    result[i, j, 0] = min(255, max(1, result[i, j, 1]) / 4 * sr)
                else:
                    result[i, j, 2] = min(255, max(1, result[i, j, 1]) / 4 * sb)
            else:
                result[i, j, 0] = min(255, max(1, G[i, j]) / 2 * sr)
                result[i, j, 2] = min(255, max(1, G[i, j]) / 2 * sb)
    
    # Add the original image to the result and convert it to uint8 data type
    return (result + img).astype(np.uint8)
