import Mascara
import os
import cv2
import numpy as np

vetor_imagens_frequencia = []

for img_atual, i in enumerate(Mascara.ROIs):
    # Aplicando FFT2D
    fft_img = np.fft.fft2(i)

    # Deslocando o centro de frequência para o meio da matriz
    fft_deslocada = np.fft.fftshift(fft_img)

    # Espectro de potência
    espectro_potencia = np.abs(fft_deslocada) ** 2

    # Aplicando log para melhor visualização
    espectro_visual = np.log(1 + espectro_potencia)

    min_val = espectro_visual.min()
    max_val = espectro_visual.max()

    if max_val - min_val > 0:
        espectro_normalizado = 255 * (espectro_visual - min_val) / (max_val - min_val)
    else: 
        espectro_normalizado = espectro_visual

    espectro_unit8 = espectro_normalizado.astype(np.uint8)

    # =========================================================================
    # ENTRADA DO MAPA DE CORES:
    # Transforma a matriz de 1 canal (escala de cinza) em 3 canais (Colorido BGR)
    # =========================================================================
    espectro_colorido = cv2.applyColorMap(espectro_unit8, cv2.COLORMAP_INFERNO)
    # =========================================================================

    # Salvando a imagem colorida (ajustado para salvar a variável 'espectro_colorido')
    caminho_fft2d = os.path.join('ImagensSaida/FFT2D', f'FFT2D_Imagen{img_atual + 1}.png')
    cv2.imwrite(caminho_fft2d, espectro_colorido)