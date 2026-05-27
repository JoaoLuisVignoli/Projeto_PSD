import Gabor_Filter
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)

theta = np.array([0, 25, 50, 75, 100, 125, 150, 175])

vetor_imagens_filtradas = []
vetor_imagens_threshold = []
vetor_imagens_filtro = []

max_img = []
min_img = []

for i in theta:
    kernel = Gabor_Filter.gabor_2d(33, i, 0.06, 9, 0.1, 0)
    filtered_img = cv2.filter2D(img, cv2.CV_32F, kernel)
    
    filtered_clip = np.clip(filtered_img, 0, None)
    
    filtered_img_norm = cv2.normalize(filtered_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    vetor_imagens_filtradas.append(filtered_img_norm)

    max_img.append(np.max(filtered_img_norm))
    min_img.append(np.min(filtered_img_norm))

    kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    vetor_imagens_filtro.append(kernel_norm)

valor_maximo = max(max_img)

for i in vetor_imagens_filtradas:
    _, thresh = cv2.threshold(i, 0.25 * valor_maximo, 255, cv2.THRESH_BINARY)

    thresh = thresh.astype(np.uint8)

    vetor_imagens_threshold.append(thresh)

print(min_img)
print(max_img)

cv2.imwrite('filtro_Gabor.tif', vetor_imagens_filtradas[3])
cv2.imwrite('Threshold_75.tif', vetor_imagens_threshold[3])