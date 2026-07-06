import Gabor_Filter
import Anisotropic_Diffusion
import os
import cv2
import numpy as np
from scipy import signal

# Vetor de ângulos
theta = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 
                  90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180])

# Matriz para guardar as imagens de saída
matriz_imagens_filtradas = []
matriz_imagens_threshold = []

# Aplicação do banco de filtros de Gabor e Threshold
for i in range(len(Anisotropic_Diffusion.semRuido)):
    linha_filtrada = []
    linha_filtrada_normalizada = []

    for j in theta:
        
        # Normalização dos valores dos pixels da imagem para o range de [0, 1]
        imagem_normalizada = Anisotropic_Diffusion.semRuido[i] / 255
        
        # Criação do Kernel do Filtro
        kernel = Gabor_Filter.gabor_2d(33, j, 0.12, 5, 0.6, 0)
        
        # Convolução da imagem com o Kernel
        filtered_img = signal.convolve2d(imagem_normalizada, kernel, mode='same')

        # Clip para remover os valores negativos
        filtered_clip = np.clip(filtered_img, 0, None)
        linha_filtrada.append(filtered_clip)

        # Salvamento das imagens dos kernels de Gabor
        if i == 0:
            kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            caminho_kernel = os.path.join('ImagensSaida/Kernel', f'Kernel_Angulo{j}.png')
            cv2.imwrite(caminho_kernel, kernel_norm)
        
    maximo_global = np.max(linha_filtrada)

    linha_threshold = []

    # Aplicando o Threshold
    for pos_atual, imagem in enumerate(linha_filtrada):
        
        # Normalização para a escala de [0, 255] para visualização da imagem
        imagem_filtrada_normalizada = ((imagem / maximo_global) * 255).astype(np.uint8)
        linha_filtrada_normalizada.append(imagem_filtrada_normalizada)
        
        # Threshold binário
        _, thresh = cv2.threshold(imagem_filtrada_normalizada, 0.3 * 255, 255, cv2.THRESH_BINARY)

        # Conversão para inteiro
        thresh = thresh.astype(np.uint8)

        linha_threshold.append(thresh)

        # Salvamento da imagem
        angulo_atual = theta[pos_atual]
        caminho_thresh = os.path.join('ImagensSaida/Threshold', f'Imagem{i+1}_Threshold_{angulo_atual}.tif')
        cv2.imwrite(caminho_thresh, thresh)

    matriz_imagens_threshold.append(linha_threshold)