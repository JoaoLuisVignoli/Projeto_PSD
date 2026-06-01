import GaborFilterBank
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

contabilizador_pixels = []

for i in GaborFilterBank.vetor_imagens_threshold:
    quantidade_pixels = np.sum(i == 255)
    contabilizador_pixels.append(quantidade_pixels)

print(len(contabilizador_pixels))
print(contabilizador_pixels)

# Diagrama de Rosas
max = [max(contabilizador_pixels) * 1.5] * 8 

df = pd.DataFrame({
    'categories': ['0', '25', '50', '75', '100', '125', '150', '175'],
    'scores': contabilizador_pixels,
    'max_values': max,
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
ax.set_yticklabels([]) # Limpa os números internos

plt.tight_layout()
print("Processamento concluído. O gráfico deve aparecer agora!")
plt.show()