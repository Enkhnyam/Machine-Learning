# With this tool you will be able to identify the best regression model for your dataset in a flashlight!
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
y_svr = y.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Training the Multiple Linear Regression model on the training set
reg_mr = LinearRegression().fit(X_train, y_train)
y_pred_mr = reg_mr.predict(X_test)
result_mr = r2_score(y_test, y_pred_mr)

# Training the Polynomial Regression model on the training set
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
reg_pr = LinearRegression().fit(X_poly, y_train)
y_pred_pr = reg_pr.predict(poly_reg.transform(X_test))
result_pr = r2_score(y_test, y_pred_pr)

# Training the Support Vector Regression(SVR) model on the training set
X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(X, y_svr, random_state=0, test_size=0.2)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svr = sc_X.fit_transform(X_train_svr)
y_train_svr = sc_y.fit_transform(y_train_svr)
reg_svr = SVR(kernel='rbf')
reg_svr.fit(X_train_svr, y_train_svr.ravel())
y_pred_svr = sc_y.inverse_transform(reg_svr.predict(sc_X.transform(X_test_svr)).reshape(-1,1))
result_svr = r2_score(y_test_svr, y_pred_svr)

# Train the Decision Tree Regression model on the training set
reg_dtr = DecisionTreeRegressor(random_state=0)
reg_dtr.fit(X_train, y_train)
y_pred_dtr = reg_dtr.predict(X_test)
result_dtr = r2_score(y_test, y_pred_dtr)

# Train the Random Forest Regression model on the training set
reg_rfr = RandomForestRegressor(n_estimators=10, random_state=0)
reg_rfr.fit(X_train, y_train)
y_pred_rfr = reg_rfr.predict(X_test)
result_rfr = r2_score(y_test, y_pred_rfr)

#Evaluating the Model Performance
print('Multiple Linear Regression Model Result: ' + str(result_mr) + '\n'+
      'Polynomial Linear Regression Model Result: ' + str(result_pr) + '\n'+
      'Support Vector Regression Model Result: ' + str(result_svr) + '\n'+
      'Decision Tree Regression Model Result: ' + str(result_dtr) + '\n' +
      'Random Forest Regression Model Result: ' + str(result_rfr)
    )
print('Winner: ' + str(max(result_dtr, result_mr, result_pr, result_rfr, result_svr)))
