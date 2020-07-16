import pandas as pd
import numpy as np
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('Wine_Quality_Data.csv')


sns.pairplot(data, hue= 'color')

y = (data['color'] == 'red').astype(int)
fields = list(data.columns[:-1])  # everything except "color"
correlations = data[fields].corrwith(y)
correlations.sort_values(inplace=True)
correlations
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation')
data.corr()



#mostly correlated variables are volatile_acidity and total_sulfur_dioxide
X = data[['volatile_acidity','total_sulfur_dioxide']].values
y=y.to_frame()
X = pd.DataFrame(X)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.svm import LinearSVC

LSVC = LinearSVC()
LSVC.fit(X, y)
X_color = X.sample(300, random_state=45)
y_color = y.loc[X_color.index]

plt.scatter(X_color[0],y_color)
plt.scatter(X_color[1],y_color)
plt.show()

