import GaborFilterBank
import Anisotropic_Diffusion
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

matriz__pixels = []

for i in range(len(Anisotropic_Diffusion.semRuido)):
    
    pixels_escala = []
    
    for s in range(len(GaborFilterBank.sigma)):
        
        pixels_angulo = []
        
        for t in range(len(GaborFilterBank.theta)):
            
            imagem = GaborFilterBank.matriz_imagens_threshold[i][s][t]
            
            quantidade_pixels = np.sum(imagem == 255)
            pixels_angulo.append(quantidade_pixels)
            
        pixels_escala.append(pixels_angulo)
        
    matriz__pixels.append(pixels_escala)
            
# Diagrama de Rosas
for img_atual, imagens in enumerate(matriz__pixels):
    
    for esc_atual, dados_escalas in enumerate(imagens):
        
        max_val = [max(dados_escalas) * 1.5] * len(GaborFilterBank.theta)
    
        df = pd.DataFrame({
            'categories': ['0', '', '', '15', '', '', '30', '', '', '45', '', '', '60', '', '', '75', '', '', '90', '', '', '105', '', '', '120', '', '', '135', '', '', '150', '', '', '165', '', '', '180'],
            'scores': dados_escalas,
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
    
        escala_atual = GaborFilterBank.sigma[esc_atual]
        ax.set_title(f'($\sigma$ = {escala_atual})', pad=20, fontsize=14, fontweight='bold')
    
        plt.tight_layout()
    
        os.makedirs(f'ImagensSaida/DiagramaDeRosas/Escala{GaborFilterBank.sigma[esc_atual]}', exist_ok=True)
        caminho_diagrama = os.path.join(f'ImagensSaida/DiagramaDeRosas/Escala{GaborFilterBank.sigma[esc_atual]}', f'Diagrama_Rosas_Imagem{img_atual + 1}.png')
        
        plt.savefig(caminho_diagrama, dpi=300, bbox_inches='tight')
        plt.close(fig)