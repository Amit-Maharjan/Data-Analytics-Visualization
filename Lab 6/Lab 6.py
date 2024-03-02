import pandas as pd
dataset = pd.read_csv('ClassificationLabData.csv')
print(dataset.head())

inputs = dataset.drop('Label', axis='columns')
target = dataset['Label']

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
inputs['worktype'] = encoder.fit_transform(inputs['worktype'])
inputs['EducationLevel'] = encoder.fit_transform(inputs['EducationLevel'])
inputs['marital_status'] = encoder.fit_transform(inputs['marital_status'])
inputs['CurrentOccupation'] = encoder.fit_transform(inputs['CurrentOccupation'])
inputs['RelationshipStatus'] = encoder.fit_transform(inputs['RelationshipStatus'])
inputs['race'] = encoder.fit_transform(inputs['race'])
inputs['Gender'] = encoder.fit_transform(inputs['Gender'])
print(inputs.head())

X = inputs
y = target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics
print('Decision Tree Confusion Matric: \n', metrics.confusion_matrix(y_test, y_pred))
print('Decision Tree Accuracy: \n', metrics.classification_report(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB
nbModel = GaussianNB()
nbModel = nbModel.fit(X_train, y_train)

y_pred = nbModel.predict(X_test)

print('Naïve Bayesian Confusion Matric: \n', metrics.confusion_matrix(y_test, y_pred))
print('Naïve Bayesian Accuracy: \n', metrics.classification_report(y_test, y_pred))