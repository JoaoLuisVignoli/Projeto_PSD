import Gabor_Filter
import Anisotropic_Diffusion
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

theta = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 
                  90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180])

matriz_imagens_filtradas = []
matriz_imagens_threshold = []
matriz_imagens_filtro = []


for i in range(len(Anisotropic_Diffusion.semRuido)):
    linha_filtrada = []
    linha_filtro = []
    linha_max = []

    for j in theta:
        kernel = Gabor_Filter.gabor_2d(33, j, 0.12, 5, 0.6, 0)
        filtered_img = cv2.filter2D(Anisotropic_Diffusion.semRuido[i], cv2.CV_32F, kernel)

        filtered_clip = np.clip(filtered_img, 0, None)

        filtered_img_norm = cv2.normalize(filtered_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        linha_filtrada.append(filtered_img_norm)

        caminho_gabor = os.path.join('ImagensSaida/Gabor', f'Imagem{i+1}_Angulo{j}.tif')
        cv2.imwrite(caminho_gabor, filtered_img_norm)

        linha_max.append(np.max(filtered_img_norm))

        kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        linha_filtro.append(kernel_norm)

    matriz_imagens_filtradas.append(linha_filtrada)
    matriz_imagens_filtro.append(linha_filtro)

    valor_maximo = max(linha_max)

    linha_threshold = []

    for pos_atual, l in enumerate(linha_filtrada):
        _, thresh = cv2.threshold(l, 0.4 * valor_maximo, 255, cv2.THRESH_BINARY)

        thresh = thresh.astype(np.uint8)

        linha_threshold.append(thresh)

        angulo_atual = theta[pos_atual]
        caminho_thresh = os.path.join('ImagensSaida/Threshold', f'Imagem{i+1}_Threshold_{angulo_atual}.tif')
        cv2.imwrite(caminho_thresh, thresh)

    matriz_imagens_threshold.append(linha_threshold)