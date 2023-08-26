import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
tips = sns.load_dataset('tips')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))

sns.barplot(x='sex', y='total_bill', data=tips, estimator=np.std)
plt.pause(0.2)
sns.countplot(x='sex', data=tips)
plt.pause(0.2)

sns.boxplot(x='day', y='total_bill', data=tips)
plt.pause(0.2)
plt.clf()
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
plt.pause(0.2)
plt.clf()
plt.figure()
# plt.fuigure() creates the new plot in new figure instead of using the same figure for the next plot

sns.violinplot(x='day', y='total_bill', data=tips)
plt.pause(0.2)
plt.clf()

# showing male female separately as violin plots
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex')
plt.pause(0.2)
plt.clf()

# comdingin the male female plot into a single viloin by splitting the violin plot into two halves of different hues
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
plt.pause(0.22)
plt.close('all')
# to create the next two plots side by side as subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
# 1 row, 2 columns
sns.stripplot(x='day', y='total_bill', data=tips, ax=axes[0])
""" 
for single rowed subplots ax=axes[i] where i is subplot number
for multi rowed subplots ax=axes[i, j] where i is row number and j is column number
for seaborn plots ax determines the location of the subplot
 """
sns.stripplot(x='day', y='total_bill', data=tips, jitter=False, ax=axes[1])
# jitter is True by default in version 3.11

sns.stripplot(x='day', y='total_bill', data=tips,
              jitter=False, hue='sex', ax=axes[2])

plt.pause(2)

plt.figure(figsize=(16, 9))
sns.violinplot(x='day', y='total_bill', data=tips)
sns.swarmplot(x='day', y='total_bill', data=tips, color='black')

plt.close('all')

sns.catplot(x='day', y='total_bill', data=tips, hue='sex', kind='bar')
# factorplot() was replaced by catplot()

# plt.show()
plt.waitforbuttonpress()
