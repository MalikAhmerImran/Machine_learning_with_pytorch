import pandas as pd
import numpy as np
from utils import standardize_features,training_testing_split
from sklearn.linear_model import Perceptron
from utils import plot_decision_regions


data_set=pd.read_csv('datasets/IRIS.csv',encoding='utf-8')
X=data_set.iloc[:,[2,3]].to_numpy()
y=data_set.iloc[:,[4]]
y=np.where(y.to_numpy().ravel()=='Iris-setosa',0,np.where(y.to_numpy().ravel()=='Iris-versicolor',1,2))
X_train,X_test,y_train,y_test=training_testing_split(X,y)
X_train_std,X_test_std=standardize_features(x_train=X_train,x_test=X_test)
model=Perceptron(eta0=0.1,random_state=1)
model.fit(X_train_std,y_train)
y_pred=model.predict(X_test_std)
print("Number of wrong predictions on test data=",(y_test!=y_pred).sum())
print("Accuracy=",model.score(X_test_std,y_test))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
y=y_combined,
classifier=model)
