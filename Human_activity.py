import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Human_Activity_Recognition_Using_Smartphones_Data.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,561]

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

y = pd.DataFrame(y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN' , strategy='most_frequent',axis=0)
imputer = imputer.fit(y)
y = imputer.transform(y)


y = pd.DataFrame(y)
y.isnull().sum()
X = pd.DataFrame(X)
X.isnull().sum()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train,y_train)
y_pred = GNB.predict(X_test)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_pred)

import seaborn as sns
ax = sns.heatmap(cf,annot=True)

X_dis = X.rank()


from sklearn.model_selection import train_test_split
X_train_dis,X_test_dis,y_train_dis,y_test_dis = train_test_split(X_dis,y,test_size = 0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train_dis,y_train_dis)
y_pred_dis = MNB.predict(X_test_dis)


from sklearn.metrics import confusion_matrix
cf_dis = confusion_matrix(y_test_dis,y_pred_dis)

import seaborn as sns
ax = sns.heatmap(cf_dis,annot=True)














