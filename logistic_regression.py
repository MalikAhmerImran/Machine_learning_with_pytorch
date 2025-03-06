import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import training_testing_split, standardize_features, plot_decision_regions


data_set = pd.read_csv("datasets/IRIS.csv", encoding="utf-8")
X = data_set.iloc[:, [2, 3]].to_numpy()
y = data_set.iloc[:, 4]
y = np.where(y == "Iris-setosa", 0, np.where(y == "Iris-virginica", 2, 1))

x_train, x_test, y_train, y_test = training_testing_split(x_samples=X, y_samples=y)
x_train_std, x_test_std = standardize_features(x_train=x_train, x_test=x_test)

lr = LogisticRegression(C=100.0, solver="lbfgs", multi_class="ovr")
lr.fit(x_train_std, y_train)

x_combine_std = np.vstack((x_train_std, x_test_std))
y_combine = np.hstack((y_train, y_test))

plot_decision_regions(X=x_combine_std, y=y_combine, classifier=lr)
