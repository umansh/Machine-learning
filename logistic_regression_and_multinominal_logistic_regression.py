import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#####################################################################
## logistic regression ##

#load data
iris = datasets.load_iris()

X = iris.data[:, :2]
print('X:',X)

y = (iris.target != 0) * 1
print('y:',y)

# plot figure
plt.figure(1)
plt.subplot(2,2,1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')

# train model
model = linear_model.LogisticRegression()
model.fit(X,y)
preds = model.predict(X)
(preds == y).mean()


#ranges plot the figure
X_min, X_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
Y_min, Y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

# define mesh grid step size & mesh grid
step_size = 0.01
X_mesh, Y_mesh = np.meshgrid(np.arange(X_min, X_max,step_size), np.arange(Y_min, Y_max,step_size))
mesh_output = model.predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()])
mesh_output = mesh_output.reshape(X_mesh.shape)

#plot figure
plt.figure(1)
plt.subplot(2,2,2)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.contour(X_mesh, Y_mesh, mesh_output, [0.5], linewidths=1, colors='black')
plt.legend()
plt.show()

############################################################################################################
## multinominal regression ##

#load data
iris = datasets.load_iris()
X = iris.data
print('X:',X)

y = iris.target
print('y:',y)

#split data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#fit model with x & y
model = linear_model.LogisticRegression(random_state = 0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print('y_test:',y_test, '\n' 'y_pred:',y_pred)
print('Accuracy:',accuracy_score(y_test,y_pred)*100)
