## Problem 1
import pandas as pd 
dataset = pd.read_csv('carsDataset.csv')

factor = 2
cols = ['mpg', 'qsec', 'hp']
outliers = {}
for col in cols:
    upperLimit = dataset[col].mean() + dataset[col].std() * factor
    lowerLimit = dataset[col].mean() - dataset[col].std() * factor
    outliers[col] = dataset[(dataset[col] >= upperLimit) | (dataset[col] <= lowerLimit)]
    print('\nThe outliers for {0} are:\n {1}'.format(col, outliers[col]))

## Problem 2
dataset = pd.read_csv('bankloan.csv')

import matplotlib.pyplot as plt
cols = ['x1', 'x5', 'x6', 'x7', 'x11', 'x13', 'x14']
for col in cols:
    boxplot = dataset[col].to_frame().dropna().boxplot()
    plt.title('Boxplot for {0}'.format(col))
    plt.show()

# Scale the data to calculate distances
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset.fillna(0))

# Create clusters from dataset
from sklearn.cluster import DBSCAN
outlier_detection  = DBSCAN(min_samples=2, eps=1.13)
clusters = outlier_detection.fit_predict(dataset_scaled)
# Trim dataframe to outliers as identified by cluster analysis
outliers = dataset.iloc[(clusters == -1).nonzero()]
print('\nTop 10 multivariate outliers:\n', outliers)