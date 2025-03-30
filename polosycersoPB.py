# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:27:58 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt


wc=1
q=1

#s1=(-wc/(2*q)) 
s1 = -0.5
#im1= (np.sqrt((wc/q)**2-4*wc**2))/2
#problema porque no me toma el nro imaginario, escribo raiz |discriminante|
im1 =(np.sqrt(3))/2


plt.figure()

circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(circle)

plt.plot([s1], [im1], 'rx', label='Polo1')
plt.plot([s1], [-im1], 'rx', label='Polo1 conjugado')  # Polo
plt.plot(0, 0, 'bo', label='Cero: (0, 0)')  # Cero

plt.xlabel('sigma')
plt.ylabel('jw')
plt.title('Diagrama de Polos y Ceros')

plt.legend()

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)


plt.grid(True)
plt.show()
