import numpy as np
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset
from utils import training_testing_split,standardization


# Loading the data set from Sklear 
iris=load_iris()
X=iris['data']
y=iris['target']

# Splitting the data set into training and testing
X_train,X_test,y_train,y_test=training_testing_split(x_samples=X,y_samples=y)

X_train_norm=standardization(X_train)
X_test_norm=standardization(X_test)
