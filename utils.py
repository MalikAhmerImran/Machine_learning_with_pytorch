import numpy as np
from sklearn.preprocessing import StandardScaler

# utility function to standardize the training and testing the samples
def standardize_features(x_train:np.ndarray,x_test:np.ndarray):
    ss=StandardScaler()
    ss.fit(x_train)
    return ss.transform(x_train),ss.transform(x_test)

    