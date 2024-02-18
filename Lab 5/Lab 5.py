import pandas as pd
data = pd.read_csv('Lab Album Sales.csv')

# Set the contants for scatter plots
import numpy as np 
colors = np.array([0.5, 0.5, 0.5])
area = np.pi*3

import matplotlib.pyplot as plt
plt.scatter(data.AdvertBudget, data.totalsales, s=area, c=colors, alpha=0.5)
plt.title('Sales Vs Advertising')
plt.xlabel('Advertising')
plt.ylabel('Sales')
plt.show()

plt.scatter(data.AirplayTimes, data.totalsales, s=area, c=colors, alpha=0.5)
plt.title('Sales Vs Airplay')
plt.xlabel('Airplay')
plt.ylabel('Sales')
plt.show()

plt.scatter(data.AttractivenessScore, data.totalsales, s=area, c=colors, alpha=0.5)
plt.title('Sales Vs Attractiveness')
plt.xlabel('Attractiveness')
plt.ylabel('Sales')
plt.show()

import statsmodels.formula.api as smf
linearModel = smf.ols(formula='totalsales ~ AdvertBudget', data=data).fit()
print(linearModel.pvalues.to_string())
print(linearModel.summary())

print('\nF-statistic and P-value for Linear Regression\nF-statistic:', linearModel.fvalue)
print('P-value:', linearModel.f_pvalue)

print('\nIntercept of Linear Regression:', linearModel.params[0])
print('\nCoefficient of Linear Regression:', linearModel.params[1])

print('\nRecords that will be sold if we spent $135,000 on advertising the latest album “Dear Agony” by Breaking Benjamin:', linearModel.params[0] + (linearModel.params[1] * 135000))

# Multiple Linear Regression
linearModelMultiple = smf.ols(formula='totalsales ~ AdvertBudget + AirplayTimes + AttractivenessScore', data=data).fit()
print(linearModelMultiple.pvalues.to_string())
print(linearModelMultiple.summary())

print('\nF-statistic and P-value for Multiple Regression\nF-statistic:', linearModelMultiple.fvalue)
print('P-value:', linearModelMultiple.f_pvalue)