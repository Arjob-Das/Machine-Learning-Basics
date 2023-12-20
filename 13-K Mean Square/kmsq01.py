from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

data = make_blobs(n_samples=200, n_features=2, centers=4,
                  cluster_std=1.8, random_state=101)

print("First element of data : \n{d}".format(d=data[0]))

print("Shape of data : \n{d}".format(d=data[0].shape))

plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

plt.pause(2)

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
print("Center of the clusters : \n{d}".format(d=kmeans.cluster_centers_))
print("Cluster labels : \n{d}".format(d=kmeans.labels_))
print("Real Labels : \n{d}".format(d=data[1]))
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
plt.pause(2)
plt.close('all')

# comparing different cluster sizes
fig, ax = plt.subplots(4, 4, sharey=True, figsize=(15, 8))

for i in range(0, 4):
    for j in range(0, 4, 2):
        kmeans = KMeans(n_clusters=i+j+1)
        kmeans.fit(data[0])

        ax[i, j].set_title("Original")
        ax[i, j].scatter(data[0][:, 0], data[0][:, 1],
                         c=data[1], cmap='rainbow')

        ax[i, j+1].set_title(str(i+j+1) + ' Clusters K means')
        ax[i, j+1].scatter(data[0][:, 0], data[0][:, 1],
                           c=kmeans.labels_, cmap='rainbow')

plt.pause(4)
plt.close('all')
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
print("Center of the clusters : \n{d}".format(d=kmeans.cluster_centers_))
print("Cluster labels : \n{d}".format(d=kmeans.labels_))
print("Real Labels : \n{d}".format(d=data[1]))
x = data[0][:, 0]
y = data[0][:, 1]
norm = plt.Normalize(y.min(), y.max())
cmap = plt.cm.viridis
sm1 = ScalarMappable(cmap=cmap, norm=norm)
sm1.set_array([])  # Set an empty array to allow using the color scale

sm2 = ScalarMappable(cmap=cmap, norm=norm)
sm2.set_array([])
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K Means')
scatter1 = ax1.scatter(x, y, c=kmeans.labels_, cmap=cmap, norm=norm)
ax2.set_title('Original')
scatter2 = ax2.scatter(x, y, c=data[1], cmap=cmap, norm=norm)
plt.show()

plt.waitforbuttonpress(0)
