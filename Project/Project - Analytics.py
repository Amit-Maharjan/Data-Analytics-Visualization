### C. Descriptive Analytics
import pandas as pd 
dataset = pd.read_csv("clean_data.csv")
print(dataset.head())
print("\n\n####################\n\n", dataset.shape, "\n\n####################\n\n")
print(dataset.info(), "\n\n####################\n")
print(dataset.isna().sum(), "\n\n####################\n")

# 5 Number Summary
import numpy as np
for column in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    five_num_sum = np.percentile(dataset[column], [0, 25, 50, 75, 100], interpolation='linear')
    print('\n{} Minimum Value = {}'.format(column, five_num_sum[0]))
    print('{} 1st Quartile = {}'.format(column, five_num_sum[1]))
    print('{} Median = {}'.format(column, five_num_sum[2]))
    print('{} 3rd Quartile = {}'.format(column, five_num_sum[3]))
    print('{} Maximum Value = {}'.format(column, five_num_sum[4]))

### D. Predictive Analytics
inputs = dataset.drop('Loan_Status', axis='columns')
target = dataset['Loan_Status']

# Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
inputs['Gender'] = encoder.fit_transform(inputs['Gender'])
inputs['Married'] = encoder.fit_transform(inputs['Married'])
inputs['Education'] = encoder.fit_transform(inputs['Education'])
inputs['Self_Employed'] = encoder.fit_transform(inputs['Self_Employed'])
inputs['Has_CreditCard'] = encoder.fit_transform(inputs['Has_CreditCard'])
inputs['Property_Area'] = encoder.fit_transform(inputs['Property_Area'])
print("\n\n####################\n\n", inputs.head(), "\n\n####################\n")

X = inputs
y = target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics
print('Decision Tree Confusion Matric: \n', metrics.confusion_matrix(y_test, y_pred), "\n\n####################\n")
print('Decision Tree Accuracy: \n', metrics.classification_report(y_test, y_pred), "\n\n####################\n")

from sklearn.naive_bayes import GaussianNB
nbModel = GaussianNB()
nbModel = nbModel.fit(X_train, y_train)

y_pred = nbModel.predict(X_test)

print('Naïve Bayesian Confusion Matric: \n', metrics.confusion_matrix(y_test, y_pred), "\n\n####################\n")
print('Naïve Bayesian Accuracy: \n', metrics.classification_report(y_test, y_pred))