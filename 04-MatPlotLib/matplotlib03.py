import matplotlib.pyplot as plt
import numpy as np
plt.ion()

x = np.linspace(0, 5, 11)
y = x**2

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, x**2, label='X squared')
axes.plot(x, x**3, label='X cubed')
axes.legend(loc=0)
axes.set_xlabel('X label')
axes.set_ylabel('Y label')
axes.set_title('Title')

plt.tight_layout()
plt.waitforbuttonpress()
