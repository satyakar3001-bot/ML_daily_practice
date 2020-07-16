import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.iloc[: , 2:12].values
y = train.iloc[:, 1].values

X_test = test.iloc[: , 1:11].values



X = np.delete(X ,8 , axis =1)
X = np.delete(X ,6 , axis =1)
X = np.delete(X ,1 , axis =1)
X_test = np.delete(X_test ,8 , axis =1)
X_test = np.delete(X_test ,6 , axis =1)
X_test = np.delete(X_test ,1 , axis =1)



from sklearn.preprocessing import Imputer
imp_mode = Imputer(missing_values = 'NaN', strategy = 'most_frequent',axis = 0)
imp_median = Imputer(missing_values = 'NaN', strategy = 'median',axis = 0)
imp_mean = Imputer(missing_values = 'NaN', strategy = 'mean',axis = 0)
imp_mean = imp_mean.fit(X[:, 2:3])
X[:, 2:3] =imp_mean.transform(X[:, 2:3])
imp_mode = imp_mode.fit(X[:, 5:6])
X[:, 5:6] =imp_mode.transform(X[:, 5:6])

imp_mode_test = Imputer(missing_values = 'NaN', strategy = 'most_frequent',axis = 0)
imp_mean_test = Imputer(missing_values = 'NaN', strategy = 'mean',axis = 0)
imp_mean_test = imp_mean_test.fit(X_test[:, 2:3])
X_test[:, 2:3] =imp_mean_test.transform(X_test[:, 2:3])
imp_mode_test = imp_mode_test.fit(X_test[:, 5:6])
X_test[:, 5:6] =imp_mode_test.transform(X_test[:, 5:6])

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[: , 1] = labelencoder.fit_transform(X[: , 1])
X[: , 6] = labelencoder.fit_transform(X[: , 6])

labelencoder_test = LabelEncoder()
X_test[: , 1] = labelencoder.fit_transform(X_test[: , 1])
X_test[: , 6] = labelencoder.fit_transform(X_test[: , 6])


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x = sc_x.fit(X[:, 0:1])
X[:, 0:1] = sc_x.transform(X[:, 0:1])

sc_x = sc_x.fit(X[:, 2:3])
X[:, 2:3] = sc_x.transform(X[:, 2:3])

sc_x = sc_x.fit(X[:, 5:6])
X[:, 5:6] = sc_x.transform(X[:, 5:6])

df_X_test = pd.DataFrame(X_test)
df_X = pd.DataFrame(X)
df_Y = pd.DataFrame(y)

sc_X_test = StandardScaler()
sc_X_test = sc_X_test.fit(X_test[:, 0:1])
X_test[:, 0:1] = sc_X_test.transform(X_test[:, 0:1])

sc_x = sc_X_test.fit(X_test[:, 2:3])
X_test[:, 2:3] = sc_X_test.transform(X_test[:, 2:3])

sc_X_test = sc_X_test.fit(X_test[:, 5:6])
X_test[:, 5:6] = sc_X_test.transform(X_test[:, 5:6])

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X,y)
y_pred =classifier.predict(X_test)

from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X,y)
y_pred2 =classifier2.predict(X_test)

from sklearn.metrics import accuracy_score
