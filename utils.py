import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# utility function to standardize the training and testing the samples
def standardize_features(x_train:np.ndarray,x_test:np.ndarray):
    ss=StandardScaler()
    ss.fit(x_train)
    return ss.transform(x_train),ss.transform(x_test)


# utility function to split the dataset into training and testing 

def training_testing_split(x_samples:np.ndarray,y_samples:np.ndarray):
    X_train,X_test,y_train,y_test=train_test_split(x_samples,y_samples,test_size=0.3,random_state=1,stratify=y_samples)
    return  X_train,X_test,y_train,y_test