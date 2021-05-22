import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load iris data
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
print(data.head())


X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf  = RandomForestClassifier(n_estimators=50)
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

# confusion_matrix, classification_report, accuracy_score
output = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",output)

output1 = classification_report(y_test, y_pred)
print("Classification Report:",output1)

output2 = accuracy_score(y_test,y_pred)
print("Accuracy:",output2)
