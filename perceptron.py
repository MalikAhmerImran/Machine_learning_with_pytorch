import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from utils import standardize_features
data_set=pd.read_csv('datasets/IRIS.csv',encoding='utf-8')
X=data_set.iloc[:,[2,3]].to_numpy()
y=data_set.iloc[:,[4]]

y=np.where(y.to_numpy().ravel()=='Iris-setosa',0,np.where(y.to_numpy().ravel()=='Iris-versicolor',1,2))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
X_train_std,X_test_std=standardize_features(x_train=X_train,x_test=X_test)

