import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
tips = sns.load_dataset('tips')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))

# sns.set_style('dark')
sns.set_style('ticks')
sns.countplot(x='sex', data=tips)
plt.pause(0.2)

# sns.despine()
# By default it removes the spines in the top and right, to remove the left and bottom use the following
# sns.despine(left=True, bottom=True)

plt.figure()
sns.reset_defaults()
sns.set_style("darkgrid")
sns.set(font="Verdana")
sns.set_context("poster")
sns.countplot(x='sex', data=tips)
plt.pause(2)
plt.close('all')

plt.figure()
sns.lmplot(x='total_bill', y='tip', height=10, aspect=1.7,
           data=tips, hue='sex', palette='spring')
plt.pause(2)


plt.waitforbuttonpress()
