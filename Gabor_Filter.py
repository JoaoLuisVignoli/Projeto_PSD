import numpy as np
import cv2

# Implementação do filtro de Gabor por meio da fórmula disponibilzida
def gabor_2d(size, angle, frequency, sigma, gamma, psi):

    theta = (angle * np.pi) / 180

    # Grade de coordenadas
    half = size // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)

    # Rotação
    x_prime = X * np.cos(theta) + Y * np.sin(theta)
    y_prime = -X * np.sin(theta) + Y * np.cos(theta)

    # Envelope gaussiano
    gaussian = np.exp(-(x_prime**2 + (gamma**2) * y_prime**2) / (2 * sigma**2))

    # Normalização
    gaussian *= gamma / (2 * np.pi * sigma**2)

    # Parte senoidal
    sinusoid = np.cos(2 * np.pi * frequency * x_prime + psi)

    # Filtro final
    gabor = gaussian * sinusoid

    return gabor

# Gera o filtro (size, theta, f, sigma, gamma, psi)
kernel = gabor_2d(33, 75, 0.08, 5, 0.4, 0)

# Carrega a imagem sem ruído
img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)

# Faz a convolução da imagem com o filtro
filtered_img = cv2.filter2D(img, -1, kernel)

# Salvar resultado
cv2.imwrite("filteredImage.tif", filtered_img)

kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imwrite('GaborKernel.png', kernel_norm)