plt.figure(figsize=(12, 8))
sns.histplot(x='loan_amnt', data=df, kde=False, bins=40)