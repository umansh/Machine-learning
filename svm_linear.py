import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set()
##############################################################################
###### svm rbf ########
# load data

iris = datasets.load_iris()
print(iris)

X = iris.data[:, :2]
print('X:',X)

y = iris.target
print('y:',y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5,shuffle=True)


model = SVC(kernel='linear')
model.fit(X_train,y_train)

def model_plot(model,X,y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    X_plot = np.c_[xx.ravel(), yy.ravel()]

    output = model.predict(X_plot)
    output = output.reshape(xx.shape)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.contourf(xx, yy, output, cmap=plt.cm.tab10, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
    plt.xlim(xx.min(), xx.max())
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SV Classifier with linear kernel')



model_plot(model, X_train, y_train)
y_pred = model.predict(X_test)
model_plot(model, X_test, y_test)

print('support vectors:',model.support_vectors_)
print('Accuracy:',accuracy_score(y_test,y_pred)*100)