import cv2
import numpy as np
import ImageLoading
""" 
# Carrega a máscara e garante que ela seja lida como escala de cinza
mascara = cv2.imread('mascara.jpeg', cv2.IMREAD_GRAYSCALE)
#mascara = cv2.imread('NovaMascara.png', cv2.IMREAD_GRAYSCALE)
img_f = cv2.imread('denoisedImage.tif', cv2.IMREAD_GRAYSCALE)

# Segurança: Redimensionar a máscara caso ela seja de tamanho diferente da imagem
# Se a máscara veio de um arquivo diferente, ela pode ter pixels de diferença
if mascara.shape != img_f.shape:
    mascara = cv2.resize(mascara, (img_f.shape[1], img_f.shape[0]))

# Segurança: Garantir que a máscara seja binária (apenas 0 e 255) e tipo uint8
# Às vezes, JPEGs criam ruídos (ex: pixels com valor 1 ou 254) que quebram a lógica
_, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

# Operação Bitwise (Agora com tipos e tamanhos garantidos)
imagem_isolada = cv2.bitwise_and(img_f, img_f, mask=mascara)

# Salva o resultado
cv2.imwrite('ROI.tif', imagem_isolada)
 """
# Vetor de ROIs
ROIs = []

for i in range(ImageLoading.mascarasCarregadas):
    mascara = ImageLoading.mascarasCarregadas[i]
    img_f = ImageLoading.imagensCarregadas[i]

    if mascara.shape != img_f.shape:
        mascara = cv2.resize(mascara, (img_f.shape[1], img_f.shape[0]))
    
    _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    imagem_isolada = cv2.bitwise_and(img_f, img_f, mask=mascara)

    ROIs.append(imagem_isolada)