<<<<<<< HEAD

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = datasets.load_diabetes()

X = data.data[:, np.newaxis, 2]

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

reg  = linear_model.LinearRegression()
reg = reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

print('Coefficients: \n', reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

=======

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = datasets.load_diabetes()

X = data.data[:, np.newaxis, 2]

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

reg  = linear_model.LinearRegression()
reg = reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

print('Coefficients: \n', reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

>>>>>>> 5e6cf97bd12103166ba725d919cb632f55cd302c
