import GaborFilterBank
import matplotlib.pyplot as plt
import numpy as np

n = 8
x = [np.random.randint(1,10) for i in range(n)]
names = ['0', '25', '50', '75', '100', '125', '150', '175']

rad = np.linspace(0, np.pi, n, endpoint=False)
width = np.pi/n

plt.figure(figsize=(10,10))
ax = plt.subplot(polar=True)
ax.bar(rad, x, width=width, color='red', linewidth=1)