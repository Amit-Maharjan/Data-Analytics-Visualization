import pandas as pd
data = pd.read_csv('C:/Users/Student99/Desktop/ETSU/2nd Semester/Data Analytics & Visualization/Lab/Lab 1/Lab 1 Data.csv');
print(data);
print(data['salary']);
print(data['education']);

print(data.loc[data['salary'] >= 127]);

from statistics import mean
print(data.loc[data['salary'] > mean(data['salary'])]);