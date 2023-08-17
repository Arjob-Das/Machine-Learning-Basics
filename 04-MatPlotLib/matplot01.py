import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 5, 11)
y = x**2
print("x:", x, "\ny:", y)

plt.plot(x, y,)
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('TItle')

""" plt.show() """
plt.clf()  # to remove autoremoval warning
plt.subplot(1, 2, 1)
plt.plot(x, y, 'r')

plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')

""" plt.show() """
plt.pause(2)  # wait time in seconds
plt.close()  # close mentioned figure or all oper figures (default is all)
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, y)
axes.set_xlabel('X label')
axes.set_ylabel('Y label')
axes.set_title('Title')

fig.show()

plt.pause(4)
plt.close()


fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])

fig.show()
plt.pause(4)
# if show is used with figure it closes automatically if that is the last statement in the program however show with plt directly needs to be closed manually or using the close() function
