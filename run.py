from mosaicage import *
from bilinear import *
from recog import *
from hue import *
from contour import *

# Load an image (replace with your own image path)
basename = "Demosaic_1"
#img = np.ones((10,10,3), dtype = np.uint8) * 255
img = cv2.imread("images/" + basename + ".bmp")
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
bsize = 16

#Metrics
MSE_bayer = MSE(img, bayer)
MSE_bilinear = MSE(img, biconv)
MSE_forme = MSE(img, forme)
MSE_hue = MSE(img, hue)
MSE_contour = MSE(img, contour)
MSE_contour2 = MSE(img, contour2)

MSE_local_bayer = MSE_local(img, bayer, 16)
MSE_local_bilinear = MSE_local(img, biconv, 16)
MSE_local_forme = MSE_local(img, forme, 16)
MSE_local_hue = MSE_local(img, hue, 16)
MSE_local_contour = MSE_local(img, contour, 16)
MSE_local_contour2 = MSE_local(img, contour2, 16)

cv2.imwrite("output/img/" + basename + "_CFA.png", cv2.cvtColor(bayer, cv2.COLOR_RGB2BGR) )
cv2.imwrite("output/img/" + basename + "_DEMO_BILINEAR_LOOP.png", cv2.cvtColor(bilinear, cv2.COLOR_RGB2BGR) )
cv2.imwrite("output/img/" + basename + "_DEMO_BILINEAR_CONV.png", cv2.cvtColor(biconv, cv2.COLOR_RGB2BGR) )
cv2.imwrite("output/img/" + basename + "_DEMO_FORME.png", cv2.cvtColor(forme, cv2.COLOR_RGB2BGR) )
cv2.imwrite("output/img/" + basename + "_DEMO_HUE.png", cv2.cvtColor(hue, cv2.COLOR_RGB2BGR) )
cv2.imwrite("output/img/" + basename + "_DEMO_CONTOUR.png", cv2.cvtColor(contour, cv2.COLOR_RGB2BGR) )
cv2.imwrite("output/img/" + basename + "_DEMO_CONTOUR2.png", cv2.cvtColor(contour2, cv2.COLOR_RGB2BGR) )

fig = plt.figure(figsize = (12,7))
columns = 3
rows = 1

fig.add_subplot(rows, columns, 1)
plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('original')

fig.add_subplot(rows, columns, 2)
plt.imshow(bayer, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('bayer')

fig.add_subplot(rows, columns, 3)
plt.imshow(biconv, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('bilinear convolution')

plt.show()

fig = plt.figure(figsize = (12,7))

fig.add_subplot(rows, columns, 1)
plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('original')

fig.add_subplot(rows, columns, 2)
plt.imshow(bayer, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('bayer')

fig.add_subplot(rows, columns, 3)
plt.imshow(forme, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('form recognition')

plt.show()

fig = plt.figure(figsize = (12,7))

fig.add_subplot(rows, columns, 1)
plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('original')

fig.add_subplot(rows, columns, 2)
plt.imshow(bayer, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('bayer')

fig.add_subplot(rows, columns, 3)
plt.imshow(hue, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('hue constancy')

plt.show()

plt.show()

fig = plt.figure(figsize = (12,7))

fig.add_subplot(2, 2, 1)
plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('original')

fig.add_subplot(2, 2, 2)
plt.imshow(bayer, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('bayer')

fig.add_subplot(2, 2, 3)
plt.imshow(contour, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('contour 1')

fig.add_subplot(2, 2, 4)
plt.imshow(contour2, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('contour 2')

plt.show()