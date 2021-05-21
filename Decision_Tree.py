import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

###################################################################
##########     Download pima indians diabetes dataset   ###########

data = pd.read_csv(path)

X = data.iloc[:,0:8]
y = data.iloc[:,8:9]

print('X:',X)
print('Y:',y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

model  = DecisionTreeClassifier()
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# confusion_matrix, classification_report, accuracy_score
output = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",output)

output1 = classification_report(y_test, y_pred)
print("Classification Report:",output1)

output2 = accuracy_score(y_test,y_pred)
print("Accuracy:",output2)

# plot figure
fig = plt.figure()
tree.plot_tree(model,filled=True, rounded=True,feature_names =list(data.columns[0:8])  ,class_names=['0','1'])
fig.savefig(path)
