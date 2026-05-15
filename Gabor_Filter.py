import numpy as np
import cv2

# Implementação do filtro de Gabor por meio da fórmula disponibilzida
def gabor_filter(size, theta, f, sigma, gamma, psi):
    # Tamanho do corpo do filtro
    x_range = np.linspace(-size, size, 2 * size + 1)
    y_range = np.linspace(-size, size, 2 * size + 1)
    x, y = np.meshgrid(x_range, y_range)

    # Rotação das coordenadas para orientar o filtro
    x_linha = x * np.cos(theta) + y * np.sin(theta)
    y_linha = -x * np.sin(theta) + y * np.cos(theta)
    
    # Parte Gaussiana (envoltória)
    gauss = np.exp(-(x_linha**2 + (gamma**2 * y_linha**2)) / (2 * sigma**2))
    
    # Parte Oscilatória (cosseno)
    oscillation = np.cos(2 * np.pi * f * x_linha + psi)
    
    return gauss * oscillation

# Gera o filtro
kernel = gabor_filter(6, -3 * np.pi / 5 , 0.14, 4.4, 0.1, 0)

# Carrega a imagem sem ruído
img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)

# Faz a convolução da imagem com o filtro
filtered_img = cv2.filter2D(img, -1, kernel)

# Salvar resultado
cv2.imwrite("filteredImage.tif", filtered_img)

# plt.figure(figsize=(6, 6))
# plt.imshow(kernel, cmap='gray')
# plt.axis('off')
# plt.show()