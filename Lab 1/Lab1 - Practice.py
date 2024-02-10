import pandas as pd

data = pd.read_csv('C:/Users/Student99/Desktop/ETSU/2nd Semester/Data Analytics & Visualization/Lab/Lab 1/Lab 1 Data.csv')

print(data);

print(data['job']);

from statistics import mean

mean(data['salary']);

nextData = data.loc[data['salary'] >= 100]

print(nextData);


