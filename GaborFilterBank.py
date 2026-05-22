import Gabor_Filter
import numpy as np
import cv2

img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)

theta = np.array([0, 25, 50, 75, 100, 125, 150, 175])

vetor_imagens_filtradas = []
vetor_imagens_filtro = []


for i in theta:
    
    kernel = Gabor_Filter.gabor_2d(33, i, 0.08, 5, 0.4, 0)
    filtered_img = cv2.filter2D(img, -1, kernel)

    vetor_imagens_filtradas.append(filtered_img)

    kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    vetor_imagens_filtro.append(kernel_norm)
