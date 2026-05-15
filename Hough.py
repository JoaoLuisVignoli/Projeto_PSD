import cv2
import numpy as np
import math

img1 = cv2.imread('Edges.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('denoisedImage.tif', cv2.IMREAD_GRAYSCALE)

cdst = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

lines = cv2.HoughLines(img1, 5, np.pi / 180, 125)

linhas_filtradas = []

if lines is not None:
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]

        angulo_graus = theta * (180 / np.pi)

        if 90 < angulo_graus < 93:
            ultima_linha_valida = i
            a = math.cos(theta)
            b = math.sin(theta)

            x0 = a * rho
            y0 = b * rho

            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            linhas_filtradas.append([pt1[0], pt1[1], pt2[0], pt2[1], rho])

            cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

# Salvar resultados
cv2.imwrite("Hough.tif", cdst)

# Define as linhas que serão utilizas para fazer a mascara
linhas_filtradas.sort(key=lambda x: x[4])

linha_superior = linhas_filtradas[0]
linha_inferior = linhas_filtradas[-1]

pontos_poligono = np.array([
    [linha_superior[0], linha_superior[1]], # Canto superior esquerdo
    [linha_superior[2], linha_superior[3]], # Canto superior direito
    [linha_inferior[2], linha_inferior[3]], # Canto inferior direito
    [linha_inferior[0], linha_inferior[1]]  # Canto inferior esquerdo
], np.int32)

# Redimensionar para o formato que o OpenCV exige para polígonos
pontos_poligono = pontos_poligono.reshape((-1, 1, 2))

# Cria a máscara preta com as mesmas dimensões da imagem original
mascara = np.zeros_like(img2)

# Preenche o polígono na máscara com branco (255)
cv2.fillPoly(mascara, [pontos_poligono], 255)

# Aplica a máscara na imagem original usando operação bitwise AND
imagem_isolada = cv2.bitwise_and(img2, img2, mask=mascara)

cv2.imwrite('mascara.tif', mascara)
cv2.imwrite('ROI.tif', imagem_isolada)