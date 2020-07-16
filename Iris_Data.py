import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import kurtosis, skew

data = pd.read_csv('Iris_Data.csv')

skew = pd.DataFrame(data.skew())
skew.columns = ['skew']
skew['too_+ve_skewed'] = skew['skew'] > .75
skew['too_-ve_skewed'] = skew['skew'] < -.75

kurtosis = pd.DataFrame(data.kurtosis())
kurtosis.columns = ['Kurtosis']
kurtosis['too_+ve']=kurtosis['Kurtosis']>0.75
kurtosis['too_ve'] = kurtosis['Kurtosis']<-0.75

from matplotlib import pyplot as plt
import numpy as np
fig,ax = plt.subplots(1,1)
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
ax.hist(a, bins = [0,25,50,75,100])
ax.set_title("histogram of result")
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('marks')
ax.set_ylabel('no. of students')
plt.show()


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
X = data.iloc[:,:-1].values
y = data.species


X_new = X.copy()
X_new['sepal_length_copy%s' % i] = X['sepal_length']
