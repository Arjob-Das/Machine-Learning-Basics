import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))
print("Entire flights dataset : \n{d}".format(d=flights))
print("Head of flights dataset : \n{d}".format(d=flights.head()))
tc = tips.corr()
print("Correlation data for tips dataset : \n{d}".format(d=tc))

sns.heatmap(tc)
plt.pause(0.2)

sns.heatmap(tc, annot=True)
plt.pause(0.2)

sns.heatmap(tc, annot=True, cmap='coolwarm')
plt.pause(0.2)
plt.close('all')

fp = flights.pivot_table(index='month', columns='year', values='passengers')
print("Pivot Table for flights dataset : \n{d}".format(d=fp))
sns.heatmap(fp)
plt.pause(0.2)
sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=0.4)
plt.pause(0.2)

sns.clustermap(fp)
plt.pause(2)

sns.clustermap(fp, cmap='coolwarm', standard_scale=1)
plt.pause(2)

plt.waitforbuttonpress()
