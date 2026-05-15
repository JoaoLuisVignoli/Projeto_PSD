import numpy as np
import cv2
 
img = cv2.imread('denoisedImage.tif')
edges = cv2.Canny(img, 30, 190, apertureSize=3, L2gradient=True)

cv2.imwrite("Edges.tif", edges)