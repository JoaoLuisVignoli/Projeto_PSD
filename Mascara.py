import os
import cv2
import numpy as np
import ImageLoading

# Vetor de ROIs
ROIs = []

# Aplicação da máscara
for i in range(len(ImageLoading.mascarasCarregadas)):
    mascara = ImageLoading.mascarasCarregadas[i]
    img_f = ImageLoading.imagensCarregadas[i]

    if mascara.shape != img_f.shape:
        mascara = cv2.resize(mascara, (img_f.shape[1], img_f.shape[0]))
    
    _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    imagem_isolada = cv2.bitwise_and(img_f, img_f, mask=mascara)

    caminho_ROI = os.path.join('ImagensSaida/ROI', f'ROI_Imagem{i + 1}.tif')
    cv2.imwrite(caminho_ROI, imagem_isolada)

    ROIs.append(imagem_isolada)