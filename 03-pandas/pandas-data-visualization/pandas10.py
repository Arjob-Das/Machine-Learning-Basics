import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
df1 = pd.read_csv('df1', index_col=0)
print("Dataframe df1 : \n{d}".format(d=df1))
print("Head of df1 : \n{d}".format(d=df1.head()))
df2 = pd.read_csv('df2')
print("Dataframe df2 : \n{d}".format(d=df2))
print("Head of df2 : \n{d}".format(d=df2.head()))


df1['A'].hist(bins=30)
plt.pause(0.2)
df1['A'].plot(kind='hist', bins=30)
plt.pause(0.2)
df1['A'].plot.hist()
plt.pause(0.2)

df2.plot.area(alpha=0.6)  # alpha controls transparency
plt.pause(0.2)

df2.plot.bar()
plt.pause(0.2)
df2.plot.bar(stacked=True)
plt.pause(0.2)
plt.close('all')

df1['A'].hist(bins=30)
plt.pause(0.2)

plt.close('all')
# mentioning x=index is redundant and causes errors
df1.plot.line(y='B', figsize=(12, 3), lw=1)
plt.pause(0.2)
plt.figure()
df1.plot.scatter(x='A', y='B', c='C')
plt.pause(0.2)
plt.figure()
# maps 3rd column using different colour scheme
df1.plot.scatter(x='A', y='B', c='C', cmap='coolwarm')
# c can represent a single colour for plotting two columns or third Column to work as colour gradient
plt.pause(0.2)
plt.close('all')
# plt.figure()
# maps 3rd column using different size which is multiplied by the constant 10 to make it more visible
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
df1.plot.scatter(x='A', y='B', s=df1['C'], ax=axes[0, 0], title='no abs')
df1.plot.scatter(x='A', y='B', s=abs(df1['C']), ax=axes[0, 1], title='abs')
df1.plot.scatter(x='A', y='B', s=abs(
    df1['C'])*10, ax=axes[1, 0], title='abs and *10')

# s represents size parameter, so the values from column c represent sizes of the points and the multiplier (constant 10) helps in standardising the visibility
# as some values in the column may be very small
# since size cannot be negative, it throws a runtime error which can be ignored or corrected with
plt.pause(2)
plt.close('all')

df2.plot.box()
plt.pause(2)
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])

plt.figure()
df.plot.hexbin(x='a', y='b', gridsize=25, cmap='coolwarm')
# gridsize represents the hexagon size in the plot. by changing the gridsize the hexagons of particular ranges are clubbed together
plt.pause(2)
plt.figure()
df2['a'].plot.kde()
plt.pause(2)
plt.figure()
df2.plot.density()
plt.pause(2)

plt.pause(0.2)
plt.waitforbuttonpress()
