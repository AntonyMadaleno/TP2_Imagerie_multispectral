import numpy as np
import matplotlib.pyplot as plt
import time

from mosaicage import *
from bilinear import *
from recog import *
from hue import *
from contour import *

test = np.full((500,500,3), 255, dtype= np.uint8)

I0 = time.time()
# Perform Bayer mosaicing on the image
bayer = bayer_mosaicage(test)
I1 = time.time()
# Perform bilinear interpolation using loops
bilinear = bilinear_loop(bayer)
I2 = time.time()
# Perform bilinear interpolation using convolution
biconv = bilinear_convolution(bayer)
I3 = time.time()
# Perform demosaicing using the defined method
forme = form_recognition(bayer)
I4 = time.time()
# Perform demosaicing using the defined method
hue = hue_constance(bayer)
I5 = time.time()
# Perform demosaicing using the defined method
contour = preservation_contour(bayer)
I6 = time.time()
# Perform demosaicing using the defined method
contour2 = preservation_contour2(bayer)
I7 = time.time()

exec_times = [I2 - I1, I3 - I2, I4 - I3, I5 - I4, I6 - I5, I7 - I6]
tickers = ["bilinear loop", "bilinear convoltion", "form recognition", "hue constancy", "contour 1", "contour 2"]

X_axis = np.arange(len(tickers))
    
fig = plt.figure(figsize= (12,7))
fig.add_subplot(1,1,1)
plt.bar(X_axis, exec_times, label = 'exec times') 
plt.xticks(X_axis, tickers) 
plt.xlabel("Method") 
plt.ylabel("Execution Time") 
plt.title("Time of executions") 
plt.show()