import Gabor_Filter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Mascara

""" img = cv2.imread('ROI.tif', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('imagemSintetica.png', cv2.IMREAD_GRAYSCALE)
 """

theta = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

matriz_imagens_filtradas = []
matriz_imagens_threshold = []
matriz_imagens_filtro = []

max_img = []
min_img = []

for i in range(Mascara.ROIs):
    linha_filtrada = []
    linha_filtro = []

    for j in theta:
        kernel = Gabor_Filter.gabor_2d(33, i, 0.12, 5, 0.6, 0)
        filtered_img = cv2.filter2D(Mascara.ROIs[j], cv2.CV_32F, kernel)

        filtered_clip = np.clip(filtered_img, 0, None)

        filtered_img_norm = cv2.normalize(filtered_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        linha_filtrada.append(filtered_img_norm)

        max_img.append(np.max(filtered_img_norm))
        min_img.append(np.min(filtered_img_norm))

        kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        linha_filtro.append(kernel_norm)

    matriz_imagens_filtradas.append(linha_filtrada)
    matriz_imagens_filtro.append(linha_filtro)

    valor_maximo = max(max_img)

    linha_threshold = []

    for i in linha_filtrada:
        _, thresh = cv2.threshold(i, 0.4 * valor_maximo, 255, cv2.THRESH_BINARY)

        thresh = thresh.astype(np.uint8)

        linha_threshold.append(thresh)

    matriz_imagens_threshold.append(linha_threshold)

""" 
# ==========================================
# PLOT 1: Imagens Filtradas (fig1, axes1)
# ==========================================
fig1, axes1 = plt.subplots(4, 4, figsize=(16, 12)) # Alterado para 4x4 e ajustado tamanho
axes1 = axes1.flatten()

for i in range(len(axes1)):
    if i < len(vetor_imagens_filtradas):
        axes1[i].imshow(vetor_imagens_filtradas[i], cmap='gray')
        axes1[i].set_title(f'Filtro Gabor - {theta[i]}°')
    axes1[i].axis('off') # Desliga os eixos inclusive dos plots vazios

fig1.tight_layout() 

# ==========================================
# PLOT 2: Imagens com Threshold (fig2, axes2)
# ==========================================
fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12)) # Alterado para 4x4 e ajustado tamanho
axes2 = axes2.flatten()

for i in range(len(axes2)):
    if i < len(vetor_imagens_threshold):
        axes2[i].imshow(vetor_imagens_threshold[i], cmap='gray')
        axes2[i].set_title(f'Threshold - {theta[i]}°')
    axes2[i].axis('off')

fig2.tight_layout() 

# ==========================================
# PLOT 3: Kernels do Filtro de Gabor (fig3, axes3)
# ==========================================
fig3, axes3 = plt.subplots(4, 4, figsize=(16, 12)) # Alterado para 4x4 e ajustado tamanho
axes3 = axes3.flatten()

for i in range(len(axes3)):
    if i < len(vetor_imagens_filtro):
        axes3[i].imshow(vetor_imagens_filtro[i], cmap='gray')
        axes3[i].set_title(f'Kernel Gabor - {theta[i]}°')
    axes3[i].axis('off')

fig3.tight_layout() 

# Exibe todas as janelas ao mesmo tempo
plt.show() """