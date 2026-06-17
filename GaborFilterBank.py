import Gabor_Filter
import Mascara
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

""" img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('imagemSintetica.png', cv2.IMREAD_GRAYSCALE)
 """

theta = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

matriz_imagens_filtradas = []
matriz_imagens_threshold = []
matriz_imagens_filtro = []

max_img = []
min_img = []

for i in range(len(Mascara.ROIs)):
    linha_filtrada = []
    linha_filtro = []

    for j in theta:
        kernel = Gabor_Filter.gabor_2d(33, j, 0.12, 5, 0.6, 0)
        filtered_img = cv2.filter2D(Mascara.ROIs[i], cv2.CV_32F, kernel)

        filtered_clip = np.clip(filtered_img, 0, None)

        filtered_img_norm = cv2.normalize(filtered_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        linha_filtrada.append(filtered_img_norm)

        caminho_gabor = os.path.join('ImagensSaida/Gabor', f'Imagem{i+1}_Angulo{j}.tif')
        cv2.imwrite(caminho_gabor, filtered_img_norm)

        max_img.append(np.max(filtered_img_norm))
        min_img.append(np.min(filtered_img_norm))

        kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        linha_filtro.append(kernel_norm)

    matriz_imagens_filtradas.append(linha_filtrada)
    matriz_imagens_filtro.append(linha_filtro)

    valor_maximo = max(max_img)

    linha_threshold = []

    for pos_atual, l in enumerate(linha_filtrada):
        _, thresh = cv2.threshold(l, 0.4 * valor_maximo, 255, cv2.THRESH_BINARY)

        thresh = thresh.astype(np.uint8)

        linha_threshold.append(thresh)

        angulo_atual = theta[pos_atual]
        caminho_thresh = os.path.join('ImagensSaida/Threshold', f'Imagem{i+1}_Threshold_{angulo_atual}.tif')
        cv2.imwrite(caminho_thresh, thresh)

    matriz_imagens_threshold.append(linha_threshold)