from skimage.color import rgb2gray
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import random as rng
from sklearn.cluster import KMeans
rng.seed(12345)

##########

PATH = "/Users/mattiaravasio/Documents/Clienti/Irce/Foto x Analisi SOFTWARE/YES"
#PATH = "/Users/mattiaravasio/Documents/Clienti/Irce/Step 1_Foto x prima analisi software/NOK"

files = []
for fname in os.listdir(PATH):
    if fname.endswith(".bmp"):
        files.append(fname)

files

##########

img = cv2.imread(PATH + '/' + files[9], 0)
img_orig = cv2.imread(PATH + '/' + files[9], 0)
plt.imshow(img, cmap='gray')

##########

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#edges = cv2.Canny(wire_thresh,0,255)
plt.imshow(thresh)

#########

def extractWire(img):
    for i in range(img.shape[0]):
        if img[i,1] < 255:
            break
    for j in range(img.shape[0]):
        if img[img.shape[0]-(j+1),1] < 255:
            k = img.shape[0]-(j+1)
            break
    return i, k, img[i:k,:]

#########

wire_x1, wire_x2, wire_thresh = extractWire(thresh)
#wire_thresh[1:20, :] = 0
plt.imshow(wire_thresh)
wire_x1
wire_x2

#########

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(wire_thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#########

plt.imshow(unknown)

#########

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

#########

markers = cv2.watershed(wire_thresh,markers)
#wire_thresh[markers == -1] = [255,0,0]

#########

contours, hierarchy = cv2.findContours(wire_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

try:
    hierarchy = hierarchy[0]
except:
    hierarchy = []

alert = False
for contour in contours[:-1]:
    (x, y, w, h) = cv2.boundingRect(contour)
    center, radius = cv2.minEnclosingCircle(contour)
    
    area_contour = cv2.contourArea(contour)
    area_rect = w * h
    area_circle = 3.14 * radius * radius #3.14 * (radius^2)
    
    #if area > 100:
    #    if w < 600:
    #        i = i + 1
    #        print(i)
    #        print(area)
    #        print(x,y,w,h)
    #        #print(contour)
    #        cv2.rectangle(img, (x, y+wire_x1), (x+w, y+wire_x1+h), (0, 0, 255), 3)
    #        cv2.circle(img, (int(center[0]), int(center[1]+wire_x1)), int(radius), (255, 0, 0), 2)
    #        X = x
    #        Y = y+wire_x1
    #        H = h
    #        W = w
    if area_contour > 70:
        #if area_rect / area_circle > 0.20:
        #if w < 600:
            #i = i + 1
            #print(i)
            #print(area_contour)
            #print(x,y,w,h)
            #print(contour)
        alert = True
        print(alert)
        cv2.rectangle(img, (x, y+wire_x1), (x+w, y+wire_x1+h), (255, 255, 0), 3)
        cv2.circle(img, (int(center[0]), int(center[1]+wire_x1)), int(radius), (255, 0, 0), 2)
        cv2.drawContours(img, contour, -1, (255, 0, 0), 3) 
        
        X = x
        Y = y+wire_x1
        H = h
        W = w
        C = (int(center[0]), int(center[1]+wire_x1))
        print(W)
        print(H)
        print(area_contour)

#cv2.imshow("mask", img)
#cv2.imshow("thresh", thresh)
#cv2.waitKey(0)

#########

sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
out_h = ndimage.convolve(thresh, sobel_horizontal, mode='reflect')
plt.imshow(thresh)

#########

plt.imshow(img, cmap = 'gray')

#########
if alert:
    plt.imshow(img)

#########
if C[0] - 200 < 0:
    limit1 = 0
    limit2 = 400
elif C[0] + 200 > 1600:
    limit1 = 1200
    limit2 = 1600
else:
    limit1 = C[0] - 200
    limit2 = C[0] + 200
if alert:
    #new_img = img_orig[Y-20:Y+H+20, X-20:X+W+20]
    #new_img = img_orig[wire_x1:wire_x2, X:X+W]
    new_img = img_orig[wire_x1:wire_x2, limit1:limit2]
    plt.imshow(new_img, cmap = 'gray')
#########
contours, hierarchy = cv2.findContours(wire_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours[:-1]:
    (x, y, w, h) = cv2.boundingRect(contour)
    center, radius = cv2.minEnclosingCircle(contour)
    
    area_contour = cv2.contourArea(contour)
    area_rect = w * h
    area_circle = 3.14 * radius * radius #3.14 * (radius^2)
    
    #if area_contour > 200:
    print(area_contour)
    print(area_rect)
    print(area_circle)
    #print(area_rect/area_circle)
    #cv2.rectangle(img, (x, y+wire_x1), (x+w, y+wire_x1+h), (0, 0, 255), 3)
    #cv2.circle(img, (int(center[0]), int(center[1]+wire_x1)), int(radius), (255, 0, 0), 2)
#########
wire_add = np.zeros((10, wire_thresh.shape[1]))
concatenate = np.concatenate((wire_add, wire_thresh), axis=0)
plt.imshow(concatenate)
#########
plt.imshow(img)
#########

