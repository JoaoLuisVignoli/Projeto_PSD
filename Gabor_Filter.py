import numpy as np
import cv2

# Implementação do filtro de Gabor por meio da fórmula disponibilzida
def gabor_2d(size, angle, frequency, sigma, gamma, psi):

    theta = ((90 - angle) * np.pi) / 180

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