import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset


transforms=transforms.Compose([transforms.ToTensor()])
path='./'
mnist_dataset=torchvision.datasets.MNIST(root=path,train=True,transform=transforms,download=True)

valid_mnist_dataset=Subset(mnist_dataset,torch.arange(10000))
train_mnist_datset=Subset(mnist_dataset,torch.arange(10000,len(mnist_dataset)))

test_mnist_data=torchvision.datasets.MNIST(root=path,train=False,transform=transforms,download=True)
