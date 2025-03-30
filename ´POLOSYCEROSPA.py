# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 10:35:31 2025

@author: iremo
"""

import matplotlib.pyplot as plt

plt.figure()

circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(circle)

plt.plot(-0.5, 0, 'rx', label='Polo: (-0.5, 0)')  # Polo
plt.plot(0, 0, 'bo', label='Cero: (0, 0)')  # Cero

plt.xlabel('sigma')
plt.ylabel('jw')
plt.title('Diagrama de Polos y Ceros')

plt.legend()

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)


plt.grid(True)
plt.show()
