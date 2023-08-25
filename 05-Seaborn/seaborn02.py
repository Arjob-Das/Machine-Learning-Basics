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

plt.close('all')

sns.boxplot(x='day', y='total_bill', data=tips)
plt.pause(2)
plt.close()
sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
plt.pause(2)


plt.show()
plt.waitforbuttonpress()
