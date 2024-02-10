import pandas as pd
data = pd.read_csv('C:/Users/Student99/Desktop/ETSU/2nd Semester/Data Analytics & Visualization/Lab/Lab 2/Salary Data - Ex2.csv');

import numpy as np
from statistics import mean
salary_five_num_sum = np.percentile(data['salary'], [0, 25, 50, 75, 100], interpolation = 'linear')
print('\nSalary Minimum Value = ', salary_five_num_sum[0])
print('Salary 1st Quartile = ', salary_five_num_sum[1])
print('Salary Median = ', salary_five_num_sum[2])
print('Salary Mean = ', mean(data['salary']))
print('Salary 3rd Quartile = ', salary_five_num_sum[3])
print('Salary Maximum Value = ', salary_five_num_sum[4])

education_five_num_sum = np.percentile(data['education'], [0, 25, 50, 75, 100], interpolation = 'linear')
print('\nEducation Minimum Value = ', education_five_num_sum[0])
print('Education 1st Quartile = ', education_five_num_sum[1])
print('Education Median = ', education_five_num_sum[2])
print('Education Mean = ', mean(data['education']))
print('Education 3rd Quartile = ', education_five_num_sum[3])
print('Education Maximum Value = ', education_five_num_sum[4])

prestige_five_num_sum = np.percentile(data['prestige'], [0, 25, 50, 75, 100], interpolation = 'linear')
print('\nPrestige Minimum Value = ', prestige_five_num_sum[0])
print('Prestige 1st Quartile = ', prestige_five_num_sum[1])
print('Prestige Median = ', prestige_five_num_sum[2])
print('Prestige Mean = ', mean(data['prestige']))
print('Prestige 3rd Quartile = ', prestige_five_num_sum[3])
print('Prestige Maximum Value = ', prestige_five_num_sum[4])

# Visualization
import matplotlib.pyplot as plt
data.hist(column='prestige')
plt.show()

data.hist(column='education')
plt.show()

data.plot.scatter(x='salary', y='education')
plt.show()

data.plot.scatter(x='education', y='prestige')
plt.show()

# Variance and SD
print('\nVariance of Salary = ', data['salary'].var())
print('Standard Deviation of Salary = ', data['salary'].std())