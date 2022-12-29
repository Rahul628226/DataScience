#Write a Python program to create multiple plots.

import matplotlib.pyplot as plt

import numpy as np

x = np.arange(0, 10, 0.02)
plt.subplot(2, 2, 1)
plt.plot(x, np.sin(x))
plt.subplot(2, 2, 2)
plt.plot(x, np.cos(x))
plt.subplot(2, 2, 3)
plt.plot(x, x)
plt.subplot(2, 2, 4)
plt.plot(x, 6-x)
plt.show()
