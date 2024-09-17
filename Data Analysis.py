import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Training_Car_Data.csv')

numerical_df = df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numerical_df.corr()

correlation_matrix.to_csv('correlation_matrix.csv')

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()