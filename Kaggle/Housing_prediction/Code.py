import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

#Data Visualisation
sns.heatmap(train_df.isnull())
sns.heatmap(test_df.isnull())

train_miss = train_df.isnull().sum()
test_miss = test_df.isnull().sum()
train_df.info()
test_df.info()

# Data Pre-processing (Missing Values)
test_df['LotFrontage'] =test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
train_df['LotFrontage'] =train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())

train_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis =1,inplace=True)
test_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis =1,inplace=True)
train_df.drop(['Id'],axis =1,inplace=True)
test_df.drop(['Id'],axis =1,inplace=True)

test_df['MSZoning']= test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

train_df['BsmtQual']= train_df['BsmtQual'].fillna(train_df['BsmtQual'].mode()[0])
test_df['BsmtQual']= test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

train_df['BsmtExposure']= train_df['BsmtExposure'].fillna(train_df['BsmtExposure'].mode()[0])
test_df['BsmtExposure']= test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])

train_df['BsmtFinType1']= train_df['BsmtFinType1'].fillna(train_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinType1']= test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])


train_df['BsmtFinType2']= train_df['BsmtFinType2'].fillna(train_df['BsmtFinType2'].mode()[0])
test_df['BsmtFinType2']= test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])


train_df['BsmtCond']= train_df['BsmtCond'].fillna(train_df['BsmtCond'].mode()[0])
test_df['BsmtCond']= test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])

test_df['FireplaceQu']= test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
train_df['FireplaceQu']= train_df['FireplaceQu'].fillna(train_df['FireplaceQu'].mode()[0])

test_df['GarageType']= test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
train_df['GarageType']= train_df['GarageType'].fillna(train_df['GarageType'].mode()[0])


test_df['GarageYrBlt']= test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mean())
train_df['GarageYrBlt']= train_df['GarageYrBlt'].fillna(train_df['GarageYrBlt'].mean())

test_df['GarageFinish']= test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
train_df['GarageFinish']= train_df['GarageFinish'].fillna(train_df['GarageFinish'].mode()[0])


test_df['GarageQual']= test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
train_df['GarageQual']= train_df['GarageQual'].fillna(train_df['GarageQual'].mode()[0])


test_df['GarageCond']= test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
train_df['GarageCond']= train_df['GarageCond'].fillna(train_df['GarageCond'].mode()[0])

test_df['MasVnrType']= test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
train_df['MasVnrType']= train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])

test_df['MasVnrArea']= test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
train_df['MasVnrArea']= train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mode()[0])


test_df['Electrical']= test_df['Electrical'].fillna(test_df['Electrical'].mode()[0])
train_df['Electrical']= train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])

test_df['Utilities']= test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']= test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']= test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinSF1']= test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mode()[0])
test_df['BsmtFinSF2']= test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mode()[0])
test_df['BsmtUnfSF']= test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mode()[0])
test_df['TotalBsmtSF']= test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mode()[0])
test_df['BsmtFullBath']= test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean())
test_df['BsmtHalfBath']= test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mean())
test_df['KitchenQual']= test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']= test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']= test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']= test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']= test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

test_df.to_csv('formulated.csv')
final_df = pd.concat([train_df,test_df],axis=0)

final_df = category_onehot_multcols(columns)
final_df = final_df.loc[:,~final_df.columns.duplicated()]

test = test_df.copy()

df_train = final_df.iloc[: 1460,:]
df_test = final_df.iloc[1460 :,:]

df_test.drop(['SalePrice'],axis=1,inplace =True)

X_train = df_train.drop(['SalePrice'],axis =1)
y_train = df_train['SalePrice']

##from sklearn.preprocessing import StandardScaler
##scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#df_test = scaler.transform(df_test)

#SK Learn training
#from sklearn.ensemble import RandomForestRegressor
#Regressor = RandomForestRegressor()

import xgboost
Regressor = xgboost.XGBRegressor()
#HyperParameter Tuning
n_estimators=[100,500,900,1100,1500]
max_depth = [2,3,5,10,15]
booster =['gbtree','gblinear']
learning_rate = [0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]
hyperparameter_grid = {
        'n_estimators' : n_estimators,
        'max_depth' : max_depth,
        'learning_rate' : learning_rate,
        'min_child_weight' : min_child_weight,
        'booster': booster,
        'base_score': base_score
        }
from sklearn.model_selection import RandomizedSearchCV
random_cv= RandomizedSearchCV(estimator=Regressor,
                              param_distributions=hyperparameter_grid,
                              cv=5,n_iter=50,scoring='neg_mean_absolute_error',n_jobs =4,
                              verbose =5,return_train_score=True,random_state=42)

random_cv.fit(X_train,y_train)
random_cv.best_estimator_
Regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=900, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
Regressor.fit(X_train,y_train)

X_train =pd.DataFrame(X_train)

y_pred = Regressor.predict(df_test)
pred =pd.DataFrame(y_pred)

sub = pd.read_csv('sample_submission.csv')
dataset = pd.concat([sub['Id'],pred],axis =1)
dataset.columns=['Id','SalePrice']
dataset.to_csv('sample_submission5.csv', index =False)

