import numpy as np
import cv2
 
img = cv2.imread('denoisedImage.tif')
edges = cv2.Canny(img,100,175)

cv2.imwrite("Edges.tif", edges)