# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataSet = pd.read_csv('Salary_Data.csv')
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# Importing the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Training the Simple Linear Regression model on the Training set

# Predicting the Test set results

# Visualising the Training set results

# Visualising the Test set results
