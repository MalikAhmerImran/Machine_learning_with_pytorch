import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset ,DataLoader

torch.manual_seed(1)
transforms=transforms.Compose([transforms.ToTensor()])
path='./'
mnist_dataset=torchvision.datasets.MNIST(root=path,train=True,transform=transforms,download=True)

valid_mnist_dataset=Subset(mnist_dataset,torch.arange(10000))
train_mnist_dataset=Subset(mnist_dataset,torch.arange(10000,len(mnist_dataset)))

test_mnist_data=torchvision.datasets.MNIST(root=path,train=False,transform=transforms,download=True)
valid_dl=DataLoader(valid_mnist_dataset,batch_size=64,shuffle=False)
train_dl=DataLoader(train_mnist_dataset,batch_size=64,shuffle=True)