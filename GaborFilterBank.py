import Gabor_Filter
import Anisotropic_Diffusion
import os
import cv2
import numpy as np
from scipy import signal

# Vetor de escalas
sigma = np.array([2, 5, 10, 20])

# Vetor de ângulos
theta = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 
                  90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180])

# Matriz para guardar as imagens de saída
matriz_imagens_filtradas = []
matriz_imagens_threshold = []
matriz_kernel_escala = []

# Aplicação do banco de filtros de Gabor e Threshold
for i in range(len(Anisotropic_Diffusion.semRuido)):
    
    escala_filtrada = []

    for s in sigma:
        
        angulos_filtrados = []
        angulos_filtrados_normalizados = []
        kernel_angulo = []
        
        for j in theta:

            # Normalização dos valores dos pixels da imagem para o range de [0, 1]
            imagem_normalizada = Anisotropic_Diffusion.semRuido[i] / 255

            # Criação do Kernel do Filtro
            kernel = Gabor_Filter.gabor_2d(33, j, 0.12, s, 0.6, 0)

            # Convolução da imagem com o Kernel
            filtered_img = signal.convolve2d(imagem_normalizada, kernel, mode='same')

            # Clip para remover os valores negativos
            filtered_clip = np.clip(filtered_img, 0, None)
            angulos_filtrados.append(filtered_clip)

            # Salvamento das imagens dos kernels de Gabor
            if i == 0:
                kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                kernel_angulo.append(kernel_norm)
                
                os.makedirs(f'ImagensSaida/kernel/Escala{s}', exist_ok=True)
                caminho_kernel = os.path.join(f'ImagensSaida/Kernel/Escala{s}', f'Kernel_Angulo{j}.png')
                cv2.imwrite(caminho_kernel, kernel_norm)
        
        maximo_global_da_escala = np.max(angulos_filtrados)
            
        for img_atual, imagem in enumerate(angulos_filtrados):
            
            # Normalização para a escala de [0, 255] para visualização da imagem
            imagem_filtrada_normalizada = ((imagem / maximo_global_da_escala) * 255).astype(np.uint8)
            angulos_filtrados_normalizados.append(imagem_filtrada_normalizada)
            
            os.makedirs(f'ImagensSaida/Gabor/Imagem{i+1}/Escala{s}', exist_ok=True)
            caminho_gabor = os.path.join(f'ImagensSaida/Gabor/Imagem{i+1}/Escala{s}', f'Gabor_Angulo{theta[img_atual]}.tif')
            cv2.imwrite(caminho_gabor, imagem_filtrada_normalizada)
    
        escala_filtrada.append(angulos_filtrados_normalizados)
    
    matriz_imagens_filtradas.append(escala_filtrada)
    matriz_kernel_escala.append(kernel_angulo)

    threshold_escala = []
    
    # Aplicando o Threshold
    for s in range(len(sigma)):
        
        threshold_angulo = []
        
        for a in range(len(theta)):
            
            imagem = escala_filtrada[s][a]
            
            # Threshold binário
            _, thresh = cv2.threshold(imagem, 0.3 * 255, 255, cv2.THRESH_BINARY)

            # Conversão para inteiro
            thresh = thresh.astype(np.uint8)

            threshold_angulo.append(thresh)

            # Salvamento da imagem
            angulo_atual = theta[a]
            escala_atual = sigma[s]
            
            os.makedirs(f'ImagensSaida/Threshold/Imagem{i+1}/Escala{escala_atual}', exist_ok=True)
            caminho_thresh = os.path.join(f'ImagensSaida/Threshold/Imagem{i+1}/Escala{escala_atual}', f'Threshold_Angulo{angulo_atual}.tif')
            cv2.imwrite(caminho_thresh, thresh)
            
        threshold_escala.append(threshold_angulo)
    
    matriz_imagens_threshold.append(threshold_escala)