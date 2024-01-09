import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# ion works in a way similar to jupiter notebook and makes the use of show() unnecessary
x = np.linspace(0, 5, 11)
y = x**2

""" axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) """
""" fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
 """
# plt.figure() or the creation of axes using add_axes is not necessary while using subplots

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))

""" j = 1
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.plot(x*np.exp(np.random.randint(i+1)), y*j)
    plt.title("Plot : {n}".format(n=i))
    j = j*np.exp(np.random.randint(1, 5))
 """
for cur in axes:
    cur.plot(x, y)

# this works only if there is 1 row, if there is more than 1 row, then it will not work as it creates a 2D array for axes and it cannot be used for iteration

# if rows are more than 1 in the subplots:

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

for i in range(len(axes)):
    for j in range(len(axes[0])):
        axes[i, j].plot(x, y)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))

axes[0].plot(x, y)
axes[0].set_title("Plot 1")

axes[1].plot(x, y)
axes[1].set_title("Plot 2")

""" manually plotting each subplot using axis array
single index works only for single row subplots
this can also be done using loop index
and
for multi row subplots [i,j] format of index works 
or for multi row subplots the previous commented out method works as well
 """

# plotting using loop for single rowed subplots

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))

for i in range(len(axes)):
    axes[i].plot(x, y)
    axes[i].set_title("Plot {n}".format(n=i))

# plotting without loop for multi rowed subplots

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), dpi=100)

axes[1, 1].plot(x, y)
axes[1, 1].set_title("Plot 5")
axes[1, 2].plot(x, y)
axes[1, 2].set_title("Plot 6")
axes[2, 1].plot(x, y)
axes[2, 1].set_title("Plot 8")
axes[2, 2].plot(x, y)
axes[2, 2].set_title("Plot 9")
plt.tight_layout()
plt.pause(1)
plt.close('all')

# plotting using loop for multi rowed subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), dpi=100)
c = 1
for i in range(len(axes)):
    for j in range(len(axes[0])):
        axes[i, j].plot(x, y)
        axes[i, j].set_title("Plot {n}".format(n=c))
        axes[i, j].set_xlabel("X{n}".format(n=c))
        axes[i, j].set_ylabel("Y{n}".format(n=c))

        c += 1

# saving a figure
plt.tight_layout()
# plt.tight_layout() is used before saving the figure to make sure the saved figure follows the shown output of tight_layout

fig.savefig('multirow_looped_subplot.png', dpi=250, bbox_inches='tight')

plt.pause(1)
plt.close('all')

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
