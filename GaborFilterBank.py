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
    kernel = Gabor_Filter.gabor_2d(33, i, 0.06, 5, 0.4, 0)
    filtered_img = cv2.filter2D(img, cv2.CV_32F, kernel)

    vetor_imagens_filtradas.append(filtered_img)

    max_img.append(np.max(filtered_img))
    min_img.append(np.min(filtered_img))

    kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    vetor_imagens_filtro.append(kernel_norm)

valor_maximo = max(max_img)

for i in vetor_imagens_filtradas:
    filtered_img_norm = cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(filtered_img_norm, 0.8 * valor_maximo, valor_maximo, cv2.THRESH_BINARY)

    thresh = thresh.astype(np.uint8)

    vetor_imagens_threshold.append(thresh)

cv2.imwrite('img_g.tif', vetor_imagens_filtradas[1])
cv2.imwrite('img_f.tif', vetor_imagens_threshold[3])
cv2.imwrite('img_d.tif', vetor_imagens_threshold[1])