import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
iris = sns.load_dataset('iris')
print("Entire iris dataset : \n{d}".format(d=iris))
print("Head of iris dataset : \n{d}".format(d=iris.head()))

sns.pairplot(iris)
plt.pause(0.2)

g = sns.PairGrid(iris)

g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.pause(0.2)

plt.close('all')

tips = sns.load_dataset('tips')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))

g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(sns.distplot, 'total_bill')
plt.pause(2)

g.map(plt.scatter, 'total_bill', 'tip')
plt.pause(2)


plt.waitforbuttonpress()
