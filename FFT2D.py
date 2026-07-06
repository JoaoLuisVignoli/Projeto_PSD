import Mascara
import os
import cv2
import numpy as np

for img_atual, i in enumerate(Mascara.ROIs):
    
    fft_img = np.fft.fft2(i)
    
    fft_deslocada = np.fft.fftshift(fft_img)
    
    espectro_potencia = np.abs(fft_deslocada) ** 2
    
    espectro_visual = np.log(1 + espectro_potencia)

    min_val = espectro_visual.min()
    max_val = espectro_visual.max()

    if max_val - min_val > 0:
        espectro_normalizado = 255 * (espectro_visual - min_val) / (max_val - min_val)
    else: 
        espectro_normalizado = espectro_visual

    espectro_unit8 = espectro_normalizado.astype(np.uint8)
 
    _, mascara = cv2.threshold(espectro_unit8, 160, 255, cv2.THRESH_BINARY)
    
    espectro_colorido = cv2.applyColorMap(espectro_unit8, cv2.COLORMAP_TURBO)
    espectro_destaque = cv2.bitwise_and(espectro_colorido, espectro_colorido, mask=mascara)
    
    altura, largura = espectro_unit8.shape
    cx, cy = largura // 2, altura // 2
    
    momentos = cv2.moments(mascara)
    
    if momentos['m00'] > 10:
        
        angulo_int_img = 0.5 * np.arctan2(2 * momentos['mu11'], momentos['mu20'] - momentos['mu02'])
        angulo_fib_img = angulo_int_img + np.pi/2
        
        comprimento = max(altura, largura)
    
        vx_int = np.cos(angulo_int_img)
        vy_int = np.sin(angulo_int_img)
        x1_int, y1_int = int(cx - vx_int * comprimento), int(cy - vy_int * comprimento)
        x2_int, y2_int = int(cx + vx_int * comprimento), int(cy + vy_int * comprimento)
        cv2.line(espectro_destaque, (x1_int, y1_int), (x2_int, y2_int), (255, 255, 0), 1)

        vx_fib = np.cos(angulo_fib_img)
        vy_fib = np.sin(angulo_fib_img)
        x1_fib, y1_fib = int(cx - vx_fib * comprimento), int(cy - vy_fib * comprimento)
        x2_fib, y2_fib = int(cx + vx_fib * comprimento), int(cy + vy_fib * comprimento)
        cv2.line(espectro_destaque, (x1_fib, y1_fib), (x2_fib, y2_fib), (255, 0, 255), 1)
        
        graus_int = np.degrees(-angulo_int_img) % 180
        graus_fib = np.degrees(-angulo_fib_img) % 180

        linhas_legenda = [
            f"Eixo Intensidade (Ciano): {graus_int:.1f} graus",
            f"Direcao da Fibra (Magenta): {graus_fib:.1f} graus"
        ]
        
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        tamanho_fonte = 0.45
        espessura_fonte = 1
        cor_texto = (255, 255, 255)
        
        y_inicial = altura - 40
        espacamento = 20
        
        for idx, linha in enumerate(linhas_legenda):
            y_atual = y_inicial + (idx * espacamento)
            cv2.putText(espectro_destaque, linha, (15, y_atual), fonte, tamanho_fonte, cor_texto, espessura_fonte, cv2.LINE_AA)
  
    caminho_fft2d = os.path.join('ImagensSaida/FFT2D', f'FFT2D_Imagem{img_atual + 1}.png')
    cv2.imwrite(caminho_fft2d, espectro_destaque)