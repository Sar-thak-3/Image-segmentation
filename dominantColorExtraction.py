# Dominant Color Extraction for Image Segmentation
# IMAGE SEGMENTATION

# Segmentation partitions an image into regions having similar visual appreance corresponding to parts of objects
# WE will try to extract most dominant 'K' colors using K-means
# We can apply K-Means with each pixel will reassigned closest of the K colors, leading to segmentation
import matplotlib.pyplot as plt
import cv2
import numpy as np

# im = cv2.imread('elephant.jpg')
im = cv2.imread('ele.jpg')
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
print(im.shape)
plt.imshow(im)
plt.show()

# Flatten each channel of the image
all_pixels = im.reshape((-1,3))
print(all_pixels.shape)

from sklearn.cluster import KMeans
dominant_colors = 4  # number of colors we want
km = KMeans(n_clusters=dominant_colors)

km.fit(all_pixels)  # trained the data

# From all that colors it find the 4 dominant colors i.e 4 type of colors -> 4 group of colors
centers = np.array(km.cluster_centers_,dtype='uint8')
print(centers)

# Plot what all these colors are these?
i = 1
plt.figure(0,figsize=(4,2))

colors = []
for each_col in centers:
    plt.subplot(1,4,i)
    plt.axis("off")
    i+=1
    colors.append(each_col)

    # Colors batch
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col
    plt.imshow(a)
plt.show()

# colors -> km.labels_
new_img = np.zeros((all_pixels.shape[0],3))
# print(new_img.shape)

for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]
new_img = new_img.reshape((im.shape))
new_img = new_img/255.0
print(new_img.shape)
plt.imshow(new_img)
plt.show()