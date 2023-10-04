import numpy as np
import matplotlib.pyplot as plt

from mosaicage import *
from bilinear import *
from recog import *
from hue import *
from contour import *

# Load an image (replace with your own image path)
basenames = ["Demosaic_1", "Demosaic_2", "Demosaic_3", "Demosaic_4", "Demosaic5"]

for basename in basenames:
    
    print("Calculating stat for " + basename + "\n")

    img = cv2.imread("images/" + basename + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform Bayer mosaicing on the image
    bayer = bayer_mosaicage(img)

    # Perform bilinear interpolation using loops
    bilinear = bilinear_loop(bayer)

    # Perform bilinear interpolation using convolution
    biconv = bilinear_convolution(bayer)

    # Perform demosaicing using the defined method
    forme = form_recognition(bayer)

    # Perform demosaicing using the defined method
    hue = hue_constance(bayer)

    # Perform demosaicing using the defined method
    contour = preservation_contour(bayer)

    # Perform demosaicing using the defined method
    contour2 = preservation_contour2(bayer)

    #taille des blocs pour le MSE_local
    bsize = 64

    #Metrics
    MSE_bayer = MSE(img, bayer)
    MSE_bilinear = MSE(img, biconv)
    MSE_forme = MSE(img, forme)
    MSE_hue = MSE(img, hue)
    MSE_contour = MSE(img, contour)
    MSE_contour2 = MSE(img, contour2)

    MSE_array = [MSE_bayer, MSE_bilinear, MSE_forme, MSE_hue, MSE_contour, MSE_contour2]
    MSE_tickers = ["MSE bayer cfa", "MSE bilinear", "MSE forme", "MSE hue", "MSE contour 1", "MSE contour 2"]

    MSE_local_bayer = MSE_local(img, bayer, bsize)
    MSE_local_bilinear = MSE_local(img, biconv, bsize)
    MSE_local_forme = MSE_local(img, forme, bsize)
    MSE_local_hue = MSE_local(img, hue, bsize)
    MSE_local_contour = MSE_local(img, contour, bsize)
    MSE_local_contour2 = MSE_local(img, contour2, bsize)

    X_axis = np.arange(len(MSE_array))
    
    fig = plt.figure(figsize= (12,7))
    fig.add_subplot(1,1,1)
    plt.bar(X_axis, MSE_array, label = 'MSE score') 
    plt.xticks(X_axis, MSE_tickers) 
    plt.xlabel("Method") 
    plt.ylabel("MSE Score") 
    plt.title("MSE Scores by method on " + basename) 
    plt.show() 

    fig = plt.figure(figsize = (12,7))
    columns = 3
    rows = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(MSE_local_bayer, cmap='magma', interpolation='nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('MSE_local bayer CFA')

    fig.add_subplot(rows, columns, 2)
    plt.imshow(MSE_local_bilinear, cmap='magma', interpolation='nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('MSE_local bilinear')

    fig.add_subplot(rows, columns, 3)
    plt.imshow(MSE_local_hue, cmap='magma', interpolation='nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('MSE_local hue constancy')

    fig.add_subplot(rows, columns, 4)
    plt.imshow(MSE_local_contour, cmap='magma', interpolation='nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('MSE_local contour 1')

    fig.add_subplot(rows, columns, 5)
    plt.imshow(MSE_local_contour2, cmap='magma', interpolation='nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('MSE_local contour 2')

    fig.add_subplot(rows, columns, 6)
    plt.imshow(MSE_local_forme, cmap='magma', interpolation='nearest', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('MSE_local form preservation')

    plt.show()