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
