import GaborFilterBank
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

contabilizador_pixels = []

# Diagrama de Rosas
maxes = [5, 5, 5, 2, 2, 10, 10, 10]
df = pd.DataFrame({
    'categories': [f'Cat_{i}' for i in range(len(maxes))],
    'scores': [random() * m for m in maxes],
    'max_values': maxes,
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