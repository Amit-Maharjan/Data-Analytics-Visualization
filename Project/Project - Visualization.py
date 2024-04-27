### B. Visualization
import pandas as pd 
dataset = pd.read_csv("clean_data.csv")
print(dataset.head())
print("\n\n####################\n\n", dataset.shape, "\n\n####################\n\n")

# Figure 1: Married column for accepted loans
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.countplot(x='Married', data=dataset[dataset['Loan_Status'] == 'Y'])
plt.title('Married Status for Accepted Loans')
plt.xlabel('Married')
plt.ylabel('Count')
plt.show()

# Figure 2: Married column for rejected loans
plt.figure(figsize=(8, 6))
sns.countplot(x='Married', data=dataset[dataset['Loan_Status'] == 'N'])
plt.title('Married Status for Rejected Loans')
plt.xlabel('Married')
plt.ylabel('Count')
plt.show()

# Figure 3: Scatter plot for ApplicantIncome and CoapplicantIncome
dataset.plot.scatter(x='ApplicantIncome', y='CoapplicantIncome')
plt.title("Applicant Income vs Coapplicant Income")
plt.xlabel("ApplicantIncome") 
plt.ylabel("CoapplicantIncome") 
plt.show()

# Figure 4: Box plot for ApplicantIncome, CoapplicantIncome, and LoanAmount
for column in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=dataset[column])
    plt.title(f'Box Plot of {column}')
    plt.show()