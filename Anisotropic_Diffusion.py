import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transform(img_gray):
    # Passa a imagem para float32
    img_float = np.array(img_gray, dtype='float32')
    
    # Transformação Logarítimica
    return np.log1p(img_float)

def exp_transform(log_image):
    # Transformação Exponencial (Retorno ao domínio original)
    return np.expm1(log_image)

def anisotropic_diffusion(img, iterations, k, refresh_rate, eq):
    # Cópia da imagem original para processamento
    denoisedImage = img.copy()

    for _ in range(iterations):
        # Cria matrizes de zeros nas 4 direções com o mesmo tamanho da matriz/imagem original
        nablaN = np.zeros_like(denoisedImage)
        nablaS = np.zeros_like(denoisedImage)
        nablaE = np.zeros_like(denoisedImage)
        nablaW = np.zeros_like(denoisedImage)

        # Cálculo da taxa de variação(nabla) em todas as direções da imagem
        nablaN[:-1,:] = np.diff(denoisedImage, axis=0)
        nablaS[1:,:] = -np.diff(denoisedImage, axis=0)
        nablaE[:,:-1] = np.diff(denoisedImage, axis=1)
        nablaW[:,1:] = -np.diff(denoisedImage, axis=1)

        # Função de condução c(|∇I|)
        if eq == 1:
            # c(|∇I|) = exp(-(∇I/k)^2)
            cN = np.exp(-(nablaN/k)**2)
            cS = np.exp(-(nablaS/k)**2)
            cE = np.exp(-(nablaE/k)**2)
            cW = np.exp(-(nablaW/k)**2)
        elif eq == 2:
            # c(|∇I|) = 1 / (1 + (∇I/k)^2)
            cN = 1. / (1. + (nablaN/k)**2)
            cS = 1. / (1. + (nablaS/k)**2)
            cE = 1. / (1. + (nablaE/k)**2)
            cW = 1. / (1. + (nablaW/k)**2)

        # Aplicação da difusão na imagem (aproximação da derivada no tempo ∂I/∂t) com a taxa de atualização
        denoisedImage += refresh_rate * (cN*nablaN + cS*nablaS + cE*nablaE + cW*nablaW)

    return denoisedImage


# Passagem para a escala de cinza
img_gray = cv2.imread('noisedImage.tif', cv2.IMREAD_GRAYSCALE)

# Transformação Logarítmica
log_img = log_transform(img_gray)

# Difusão Anisotrópica
ans_log_img = anisotropic_diffusion(log_img, iterations=20, k=30, refresh_rate=0.1, eq=1)

# Transformação Exponencial (Retorno ao domínio original)
ans_img = exp_transform(ans_log_img)

# Ajuste de escala final para exibição e salvamento (uint8)
ans_img = np.clip(ans_img, 0, 255).astype(np.uint8)

# Visualização dos resultados
plt.figure(figsize=(40, 20))

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img_gray, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title('Transformação Logarítimica + Difusão Anisotrópica')
plt.imshow(ans_img, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()