import pandas as pd 
from apyori import apriori

dataset = pd.read_csv("TitanicData.csv")
print('Checking the data to see if the header is aligned as it should have been:\n', dataset)

for col in dataset.columns:
    print('\nUnique values for {0}:'.format(col))
    for val in dataset[col].unique():
        print(val)

# Check if there is any missing values
print('\nTotal number of missing values:\n' + str(dataset.isna().sum()))

association_rules = apriori(dataset.values, min_support = 0.005, min_confidence = 0.8, min_length = 2)
association_results = list(association_rules)

# Filtering our results
filtered_results = []
for result in association_results:
    for entry in result.ordered_statistics:
        if(entry.items_add == frozenset({'Yes'})):
            filtered_results.append(entry)

print('\nTotal number of rules: {0}\n'.format(len(filtered_results)))

sorted_result = sorted(filtered_results, key=lambda x : x.lift, reverse = True)

for result in sorted_result:
    print(str(result))