import os
import cv2
import numpy as np

numero_imagens = 5

# Criando um vetor de imagens e máscaras
imagensCarregadas = []
mascarasCarregadas = []

# Carregando as imagens e mascaras para dentro dos vetores
for i in range(numero_imagens):
    caminho_imagens = os.path.join('ImagensEntrada/Imagens', f'Imagem{i+1}.tif')
    caminho_mascaras = os.path.join('ImagensEntrada/Mascaras', f'Mascara{i+1}.tif')
    
    imagem_carregada = cv2.imread(caminho_imagens, cv2.IMREAD_GRAYSCALE)
    mascara_carregada = cv2.imread(caminho_mascaras, cv2.IMREAD_GRAYSCALE)
    
    imagensCarregadas.append(imagem_carregada)
    mascarasCarregadas.append(mascara_carregada)