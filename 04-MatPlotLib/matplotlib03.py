import matplotlib.pyplot as plt
import numpy as np
plt.ion()

x = np.linspace(0, 5, 11)
y = x**2

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, x**2, label='X squared', color='#2F8B01', linewidth=2, alpha=0.3)
# alpha controls line opacity
# lw=2 is same as linewidth=2
axes.plot(x, x**3, label='X cubed', marker='o', color='blue', markerfacecolor='red',
          markeredgecolor='black', markeredgewidth=1.2, linewidth=0.5, markersize=5, linestyle='--')

axes.set_xlim([0, 4])
axes.set_ylim([0, 4])

axes.legend(loc=0)
axes.set_xlabel('X label')
axes.set_ylabel('Y label')
axes.set_title('Title')

plt.tight_layout()
plt.waitforbuttonpress()
