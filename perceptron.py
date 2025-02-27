import pandas as pd
import numpy as np
from utils import standardize_features,training_testing_split
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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

def plot_decision_regions(X,y,classifier,resolution=0.02):
    # markers=('o','s','>','<','^')
    # colours=('red','blue','lightgreen','grey','cyan')
    # cmap=ListedColormap(colours[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min() -1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min() -1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    lab=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    lab=lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
     # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.8,
        label=f'Class {cl}',
        edgecolor='black')
    plt.show()

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
y=y_combined,
classifier=model)
