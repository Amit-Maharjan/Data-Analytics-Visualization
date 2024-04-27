### A. Preprocessing
import pandas as pd 
dataset = pd.read_csv("bank_loan_approval_dataset.csv")
print(dataset.head())
print("\n\n####################\n\n", dataset.shape, "\n\n####################\n\n")
print(dataset.info(), "\n\n####################\n")
print(dataset.isna().sum(), "\n\n####################\n")

for column in dataset.columns:
    print('Unique values for', column, 'column:', dataset[column].unique())

print("\n####################\n\nNumber of unique value in Loan_ID column: ", len(dataset['Loan_ID'].unique()))
print("Number of unique value in Applicant_ID column: ", len(dataset['Applicant_ID'].unique()), "\n\n####################\n")

# Drop columns: 'Loan_ID', and 'Applicant_ID'
dataset = dataset.drop(columns = ['Loan_ID', 'Applicant_ID'])

# Replace the string "Unknown" to NAN value
import numpy as np 
dataset.replace("Unknown", np.nan, inplace=True)
print(dataset.isna().sum(), "\n\n####################\n")

# The column Owns_Car has 7484 out of 10000 missing values
# Droping the column
dataset = dataset.drop(columns = ['Owns_Car'])

for column in dataset.columns:
    print('Unique values for', column, 'column:', dataset[column].unique())

print("\n\n####################\n\n")
print(dataset.info(), "\n\n####################\n")

# The data type of the columns: ApplicantIncome and CoapplicantIncome are object
# Converting it into int
dataset['ApplicantIncome'] = pd.to_numeric(dataset['ApplicantIncome'], errors='coerce')
dataset['CoapplicantIncome'] = pd.to_numeric(dataset['CoapplicantIncome'], errors='coerce')
print(dataset.info(), "\n\n####################\n")
print(dataset.isna().sum(), "\n\n####################\n")

# Calculate variance
numeric_columns = dataset.select_dtypes(include=['number'])
variance = numeric_columns.var()
# Set display options to show floating-point numbers in a more human-readable format
pd.set_option('display.float_format', lambda x: '%.6f' % x)
print(variance, "\n\n####################\n")

# Calculate Pearson's correlation coefficient
print(numeric_columns.corr(), "\n\n####################\n")

# Imputing data for numeric columns: ApplicantIncome and CoapplicantIncome
from sklearn.impute import KNNImputer
column_names = ['ApplicantIncome', 'CoapplicantIncome']
imputer = KNNImputer()
dataset[column_names] = imputer.fit_transform(dataset[column_names])
print(dataset.isna().sum(), "\n\n####################\n")

# Outliers: ApplicantIncome
q1 = dataset['ApplicantIncome'].quantile(0.25)
q3 = dataset['ApplicantIncome'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
clean_data = dataset[(dataset['ApplicantIncome'] >= lower_bound) & (dataset['ApplicantIncome'] <= upper_bound)]
print(clean_data, "\n\n####################\n")

# Outliers: CoapplicantIncome
q1 = clean_data['CoapplicantIncome'].quantile(0.25)
q3 = clean_data['CoapplicantIncome'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
clean_data = clean_data[(clean_data['CoapplicantIncome'] >= lower_bound) & (clean_data['CoapplicantIncome'] <= upper_bound)]
print(clean_data, "\n\n####################\n")

# Clean file dumped 
clean_data.to_csv('clean_data.csv', index=False)