import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset,DataLoader
from utils import training_testing_split,standardization
from torch import nn

# Loading the data set from Sklear 
iris=load_iris()
X=iris['data']
y=iris['target']

# Splitting the data set into training and testing
X_train,X_test,y_train,y_test=training_testing_split(x_samples=X,y_samples=y)

X_train_norm=standardization(X_train)
X_train_norm=torch.from_numpy(X_train_norm)
y_train=torch.from_numpy(y_train)
train_ds=TensorDataset(X_train_norm,y_train)
torch.manual_seed(1)
train_dl=DataLoader(train_ds,batch_size=2,shuffle=True)


# Creating the neural network for traing the iris data set on multilayer perceptron model
class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.Layer1=nn.Linear(input_size,hidden_size)
        self.Layer2=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x=self.Layer1(x)
        x=nn.Sigmoid()(x)
        x=self.Layer2(x)
        x=nn.Softmax(dim=1)(x)
        return x
    
model=Model(input_size=X_train_norm.shape[1],hidden_size=16,output_size=3)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
num_epochs = 100
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs
for epoch in range(num_epochs):
    for x_batch,y_batch in train_dl:
        x_batch=x_batch.float()
        pred=model(x_batch)
        loss=loss_fn(pred,y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[epoch] += loss.item()*y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist[epoch] += is_correct.mean()
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

# saving the model weights
path = 'iris_classifier.pt'
torch.save(model,path)