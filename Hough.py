import cv2
import numpy as np
import math

img = cv2.imread('Edges.tif', cv2.IMREAD_GRAYSCALE)

cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(img, 0.25, np.pi / 180, 70, 40, 40)

if lines is not None:
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]

        a = math.cos(theta)
        b = math.sin(theta)

        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

# Salvar resultados
cv2.imwrite("Hough.tif", cdst)
cv2.imwrite("HoughP.tif", cdstP)