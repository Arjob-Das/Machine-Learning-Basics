from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
tips = sns.load_dataset("tips")
print("Entire tips dataset : \n{d}".format(d=tips))
print("Head of tips dataset : \n{d}".format(d=tips.head()))
sns.displot(tips["total_bill"], kde=True)
# distplot was replaced with displot
# kde=True must be used in case of displot() as it doesn't show the kde line of distplot() by default
# kde for displot() is False by default

plt.pause(0.2)

sns.displot(data=tips, x="total_bill", kde=False, bins=100)
plt.pause(0.2)

sns.jointplot(x="total_bill", y="tip", data=tips)
# by default it shows a scatter plot as the main plot
plt.pause(0.2)
sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex")
plt.pause(0.2)
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")
plt.pause(0.2)
sns.jointplot(x="total_bill", y="tip", data=tips, kind="kde")
plt.pause(0.2)

sns.pairplot(tips)
plt.pause(0.2)

sns.pairplot(tips, hue="sex", palette="coolwarm")
# hue basically categorises data based on the categorical variable
# using hue we can categorise the data based on the categorical variable
# palette is used to change the colour of the plot
# using hue changes the histogram plot of same variable to an area plot
plt.pause(0.2)
plt.close("all")

sns.rugplot(data=tips, x="total_bill")
plt.pause(0.2)
plt.close("all")

# kde means kernel density estimation plotd

# copied from notebook
# Create dataset
dataset = np.random.randn(25)

# Create another rugplot
sns.rugplot(dataset)

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min, x_max, 100)

# Set up the bandwidth, for info on this:
url = "http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth"

bandwidth = ((4 * dataset.std() ** 5) / (3 * len(dataset))) ** 0.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)

    # Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * 0.4
    plt.plot(x_axis, kernel, color="grey", alpha=0.5)

plt.ylim(0, 1)
plt.pause(0.2)
plt.close("all")  # if not closed the next plot gets the same limits as this one
# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list, axis=0)

# Plot figure
fig = plt.plot(x_axis, sum_of_kde, color="indianred")

# Add the initial rugplot
sns.rugplot(dataset, c="indianred")

# Get rid of y-tick marks
plt.yticks([])
plt.ylim(0)  # to ensure kde line starts and ends in 0
# Set title
plt.suptitle("Sum of the Basis Functions")

plt.pause(2)

plt.show()
plt.waitforbuttonpress()
