import GaborFilterBank
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

matriz__pixels = []

for linha in GaborFilterBank.matriz_imagens_threshold:
    linha_contabilizadora = []
    
    for imagem in linha:
        quantidade_pixels = np.sum(imagem == 255)
        linha_contabilizadora.append(quantidade_pixels)
        
    matriz__pixels.append(linha_contabilizadora)

# Diagrama de Rosas
for img_atual, dados in enumerate(matriz__pixels): 
    max_val = [max(dados) * 1.5] * len(GaborFilterBank.theta)

    df = pd.DataFrame({
        'categories': ['0', '15', '30', '45', '60', '75', '90', '105', '120', '135', '150', '165', '180'],
        'scores': dados,
        'max_values': max_val,
    })
    df['pct'] = df['scores'] / df['max_values']

    N = len(df)
    theta = np.linspace(0, np.pi, N, endpoint=False)
    width = np.pi / N 

    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

    bars = ax.bar(
        theta + width/2, 
        df['pct'], 
        width=width, 
        edgecolor='black', 
        color='skyblue', 
        alpha=0.7
    )

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(theta + width/2)
    ax.set_xticklabels(df['categories'])
    ax.set_yticklabels([])

    plt.tight_layout()

    caminho_diagrama = os.path.join('ImagensSaida/DiagramaDeRosas', f'Diagrama_Rosas_Imagem{img_atual + 1}.png')
    plt.savefig(caminho_diagrama, dpi=300, bbox_inches='tight')