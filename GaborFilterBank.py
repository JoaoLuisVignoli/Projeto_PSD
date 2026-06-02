import Gabor_Filter
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

theta = np.array([0, 25, 50, 75, 100, 125, 150, 175])

vetor_imagens_filtradas = []
vetor_imagens_threshold = []
vetor_imagens_filtro = []

max_img = []
min_img = []

for i in theta:
    kernel = Gabor_Filter.gabor_2d(33, i, 0.06, 9, 0.1, 0)
    filtered_img = cv2.filter2D(img, cv2.CV_32F, kernel)
    
    filtered_clip = np.clip(filtered_img, 0, None)
    
    filtered_img_norm = cv2.normalize(filtered_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    vetor_imagens_filtradas.append(filtered_img_norm)

    max_img.append(np.max(filtered_img_norm))
    min_img.append(np.min(filtered_img_norm))

    kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    vetor_imagens_filtro.append(kernel_norm)

valor_maximo = max(max_img)

for i in vetor_imagens_filtradas:
    _, thresh = cv2.threshold(i, 0.4 * valor_maximo, 255, cv2.THRESH_BINARY)

    thresh = thresh.astype(np.uint8)

    vetor_imagens_threshold.append(thresh)


# ==========================================
# PLOT 1: Imagens Filtradas (fig1, axes1)
# ==========================================
fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
axes1 = axes1.flatten()

for i in range(len(vetor_imagens_filtradas)):
    axes1[i].imshow(vetor_imagens_filtradas[i], cmap='gray')
    axes1[i].set_title(f'Filtro Gabor - {theta[i]}°')
    axes1[i].axis('off')

fig1.tight_layout() # Organiza apenas a janela 1

# ==========================================
# PLOT 2: Imagens com Threshold (fig2, axes2)
# ==========================================
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
axes2 = axes2.flatten()

for i in range(len(vetor_imagens_threshold)):
    axes2[i].imshow(vetor_imagens_threshold[i], cmap='gray')
    axes2[i].set_title(f'Threshold - {theta[i]}°')
    axes2[i].axis('off')

fig2.tight_layout() # Organiza apenas a janela 2

# ==========================================
# PLOT 3: Kernels do Filtro de Gabor (fig3, axes3)
# ==========================================
fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
axes3 = axes3.flatten()

for i in range(len(vetor_imagens_filtro)):
    # Plota os kernels armazenados no vetor correspondente
    axes3[i].imshow(vetor_imagens_filtro[i], cmap='gray')
    axes3[i].set_title(f'Kernel Gabor - {theta[i]}°')
    axes3[i].axis('off')

fig3.tight_layout() # Organiza a janela 3

# Exibe ambas as janelas ao mesmo tempo
plt.show()