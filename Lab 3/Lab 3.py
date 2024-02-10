import pandas as pd
data = pd.read_csv('C:/Users/Student99/Desktop/ETSU/2nd Semester/Data Analytics & Visualization/Lab/Lab 3/Salary Data - Ex3.csv')

import numpy as np
covariance_salary_education = np.cov(data['salary'], data['education'])[0, 1]
print('Covariance for salary and education = ', covariance_salary_education)

covariance_salary_prestige = np.cov(data['salary'], data['prestige'])[0, 1]
print('Covariance for salary and prestige = ', covariance_salary_prestige)

covariance_education_prestige = np.cov(data['education'], data['prestige'])[0, 1]
print('Covariance for education and prestige = ', covariance_education_prestige)

from scipy import stats
r_salary_education, p_salary_education = stats.pearsonr(data['salary'], data['education'])
print('\nPearson\'s correlation coefficient for salary and education = ', r_salary_education)
print('P-value for salary and education = ', p_salary_education)

r_salary_prestige, p_salary_prestige = stats.pearsonr(data['salary'], data['prestige'])
print('Pearson\'s correlation coefficient for salary and prestige = ', r_salary_prestige)
print('P-value for salary and prestige = ', p_salary_prestige)

r_education_prestige, p_education_prestige = stats.pearsonr(data['education'], data['prestige'])
print('Pearson\'s correlation coefficient for education and prestige = ', r_education_prestige)
print('P-value for education and prestige = ', p_education_prestige)