import numpy as np
import cv2

# Implementação do filtro de Gabor por meio da fórmula disponibilzida
def gabor_filter(x_size, y_size, theta, f, sigma, gamma, psi):

    # Tamanho do corpo do filtro
    x_range = np.linspace(-(x_size - 1)/2, (x_size - 1)/2, x_size)
    y_range = np.linspace(-(y_size - 1)/2, (y_size - 1)/2, y_size)
    x, y = np.meshgrid(x_range, y_range)

    # Rotação das coordenadas para orientar o filtro
    x_linha = x * np.cos(theta) + y * np.sin(theta)
    y_linha = -x * np.sin(theta) + y * np.cos(theta)
    
    # Parte Gaussiana (envoltória)
    gauss = np.exp(-(x_linha**2 + (gamma**2 * y_linha**2)) / (2 * sigma**2))
    
    # Parte Oscilatória (cosseno)
    oscillation = np.cos(2 * np.pi * f * x_linha + psi)
    
    return gauss * oscillation

# Gera o filtro (x_size, y_size, theta, f, sigma, gamma, psi)
kernel = gabor_filter(33, 33, -np.pi , 0.18, 5.5, 0.35, 0)

# Carrega a imagem sem ruído
img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)

# Faz a convolução da imagem com o filtro
filtered_img = cv2.filter2D(img, -1, kernel)

# Salvar resultado
cv2.imwrite("filteredImage.tif", filtered_img)

kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imwrite('GaborKernel.png', kernel_norm)