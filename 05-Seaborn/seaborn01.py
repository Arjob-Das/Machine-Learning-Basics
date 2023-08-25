import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
tips=sns.load_dataset('tips')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))
sns.displot(tips['total_bill'],kde=True)
#distplot was replaced with displot
#kde=True must be used in case of displot() as it doesn't show the kde line of distplot() by default
#kde for displot() is False by default

plt.pause(0.2)

sns.displot(data=tips,x='total_bill',kde=False,bins=100)
plt.pause(0.2)

sns.jointplot(x='total_bill',y='tip',data=tips)
#by default it shows a scatter plot as the main plot
plt.pause(0.2)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
plt.pause(0.2)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
plt.pause(0.2)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
plt.pause(0.2)

sns.pairplot(tips)
plt.pause(0.2)

sns.pairplot(tips,hue='sex',palette='coolwarm')
#hue basically categorises data based on the categorical variable
#using hue we can categorise the data based on the categorical variable
#palette is used to change the colour of the plot
#using hue changes the histogram plot of same variable to an area plot
plt.pause(0.2)
plt.close('all')

sns.rugplot(data=tips,x='total_bill')
plt.pause(2)
plt.close('all')

#kde means kernel density estimation plot


plt.show()
plt.waitforbuttonpress()