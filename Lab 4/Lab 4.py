import pandas as pd
data = pd.read_csv('Income Dirty Data.csv', index_col=0) # index_col=0 indicate that the values in the first column of the CSV file will be used as the row labels in the DataFrame

# Checking missing values for each column
missingValues = data.isna().sum()
print('Missing Values:')
print(missingValues)

# 1. Number and Percentage of observations that are complete
numberOfCell = len(data) * len(data.columns)
print('\nNumber of cells that are complete = ', numberOfCell - missingValues.sum())
print('Percentage of cells that are complete = ', (numberOfCell - missingValues.sum())/numberOfCell*100, '%')

numberOfRowsWithMissingValues = len(data[(data['gender'].isna()) | (data['income'].isna()) | (data['tax (15%)'].isna())]) # The column 'age' is not mentioned because it has 0 missing values
print('\nNumber of rows that are complete = ', len(data) - numberOfRowsWithMissingValues)
print('Percentage of rows that are complete = ', (len(data) - numberOfRowsWithMissingValues)/len(data)*100, '%')

# 2. Checking with the rules
validRows = len(data[(data['age'] > 18) & (data['income'] > 0) & (data['tax (15%)'] == 0.15 * data['income'])])
print('\nPercentage of data that has no errors = ', validRows/len(data)*100, '%\n')

# 3. Correcting
# Gender
print(data['gender'].unique()) # Checking the values for 'gender' column
data['gender'] = data['gender'].apply(lambda gender: 'Male' if(gender == 'Man' or gender == 'Men') else ('Female' if(gender == 'Women' or gender == 'Woman') else gender))
print(data['gender'].unique())

# Age, Income, Tax
import numpy as np 
data.loc[~(data['age'] > 0), 'age'] = np.NAN
data.loc[~(data['income'] > 0), 'income'] = np.NAN
data.loc[~(data['tax (15%)'] > 0), 'tax (15%)'] = np.NAN

data['income'] = np.where(data['income'].isna() & data['tax (15%)'].notnull(), data['tax (15%)']*(100/15), data['income'])
data['tax (15%)'] = np.where(data['tax (15%)'].isna() & data['income'].notnull(), data['income']*0.15, data['tax (15%)'])

# 4. Imputing (inserting missing values)
# Set display options to show floating-point numbers in a more human-readable format
pd.set_option('display.float_format', lambda x: '%.6f' % x)
print(data.describe())

# Check missing values again
print('Missing Values:')
print(data.isna().sum())

# Fill 'Male' or 'Female' value randomly for 'gender' column if the value is missing
data['gender'] = data['gender'].fillna(pd.Series(np.random.choice(['Male', 'Female'], size=len(data.index))))

# Encode 'gender' column with Male and Female value to numeric value 0 and 1 for ML algorithm
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])

# Rescaling all the features i.e. 'gender', 'age', 'income', and 'tax (15%)'
scaler = preprocessing.StandardScaler()
features = [['gender', 'age', 'income', 'tax (15%)']]
for feature in features:
    data[feature] = scaler.fit_transform(data[feature])

# Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer()
filledData = imputer.fit_transform(data)

# Formating the data to look like it was initially
cols = ['gender', 'age', 'income', 'tax (15%)']
finalData = pd.DataFrame(data = filledData, columns = cols)
print(finalData.describe())

# Check missing values again
print('Missing Values:')
print(finalData.isna().sum())