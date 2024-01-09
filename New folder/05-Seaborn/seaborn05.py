import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
tips = sns.load_dataset('tips')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))

sns.lmplot(x='total_bill', y='tip', data=tips)
# lmplot stands for linear model plot
plt.pause(0.2)


sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex',
           markers=['o', 'v'], scatter_kws={'s': 100})
# the dictionary {s:100} is a matplotlib kws_arg for matplotlib which maps the size of the markers in the plot in this case
plt.pause(0.2)

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row='time')
# adding row and column basically creates the subplots in the same figure
plt.pause(0.2)
sns.lmplot(x='total_bill', y='tip', data=tips,
           col='day', row='time', hue='sex')
# adding row and column basically creates the subplots in the same figure and adding hue with this farther divides the subplots
plt.pause(0.2)
plt.close('all')
sns.lmplot(x='total_bill', y='tip', data=tips, col='day',
           row='time', hue='sex', aspect=1.7, height=8)
# height controls the height of each subplot and width is calculated based on the aspect (aspect ratio)
plt.pause(2)


plt.waitforbuttonpress()
