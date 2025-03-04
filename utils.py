import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# utility function to standardize the training and testing the samples
def standardize_features(x_train:np.ndarray,x_test:np.ndarray):
    ss=StandardScaler()
    ss.fit(x_train)
    return ss.transform(x_train),ss.transform(x_test)


# utility function to split the dataset into training and testing 
def training_testing_split(x_samples:np.ndarray,y_samples:np.ndarray):
    X_train,X_test,y_train,y_test=train_test_split(x_samples,y_samples,test_size=0.3,random_state=1,stratify=y_samples)
    return  X_train,X_test,y_train,y_test


def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('o','s','>','<','^')
    colours=('red','blue','lightgreen','grey','cyan')
    cmap=ListedColormap(colours[:len(np.unique(y))])
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
        c=colours[idx],
        marker=markers[idx],
        label=f'Class {cl}',
        edgecolor='black',)
    plt.legend(loc='upper left')
    plt.show()