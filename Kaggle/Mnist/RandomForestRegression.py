import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.iloc[: , 1:].values
y = train.iloc[:, 0].values

df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.1)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)