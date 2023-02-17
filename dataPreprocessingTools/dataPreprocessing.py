# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataSet = pd.read_csv('Data.csv')
matrixOfFeatures = dataSet.iloc[:, :-1].values
dependentVariableVector = dataSet.iloc[:, -1].values

# Taking care of missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(matrixOfFeatures[:, 1:3])
matrixOfFeatures[:, 1:3] = imputer.transform(matrixOfFeatures[:, 1:3])

# Encoding categorical data

  # Encoding the Independent Variable
  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
matrixOfFeatures = np.array(ct.fit_transform(matrixOfFeatures))

  # Encoding the Dependent Variable
  
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dependentVariableVector = le.fit_transform(dependentVariableVector)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
matrixOfFeatures_train, matrixOfFeatures_test, dependentVariableVector_train, dependentVariableVector_test = train_test_split(matrixOfFeatures, dependentVariableVector, test_size=0.2, random_state=0)

# Feature Scaliing

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
matrixOfFeatures_train[:, 3:] = sc.fit_transform(matrixOfFeatures_train[:, 3:])
matrixOfFeatures_test[:, 3:] = sc.transform(matrixOfFeatures_test[:, 3:])
