import cv2
import numpy as np

# 1. Carrega a máscara e garante que ela seja lida como escala de cinza
mascara = cv2.imread('mascara.jpeg', cv2.IMREAD_GRAYSCALE)
img_f = cv2.imread('denoisedImage.tif', cv2.IMREAD_GRAYSCALE)

# 2. Segurança: Redimensionar a máscara caso ela seja de tamanho diferente da imagem
# Se a máscara veio de um arquivo diferente, ela pode ter pixels de diferença
if mascara.shape != img_f.shape:
    mascara = cv2.resize(mascara, (img_f.shape[1], img_f.shape[0]))

# 3. Segurança: Garantir que a máscara seja binária (apenas 0 e 255) e tipo uint8
# Às vezes, JPEGs criam ruídos (ex: pixels com valor 1 ou 254) que quebram a lógica
_, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

# 4. Operação Bitwise (Agora com tipos e tamanhos garantidos)
imagem_isolada = cv2.bitwise_and(img_f, img_f, mask=mascara)

# 5. Salva o resultado
cv2.imwrite('ROI.tif', imagem_isolada)