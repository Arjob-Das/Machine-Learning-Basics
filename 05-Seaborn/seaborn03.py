import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
tips = sns.load_dataset('tips')
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))
