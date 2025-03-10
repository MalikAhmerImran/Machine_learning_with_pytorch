import torch
import logging
import torchvision
from torch.utils.data import DataLoader, Dataset
from itertools import islice
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")

torch.manual_seed(1)

# creating the pytorch tensors
a = [1, 2, 3]
t_a = torch.tensor(a)
logging.info(f"Pytorch tensors: {t_a}")

# creating the pytorch tensors with values 1
t_ones = torch.ones(2, 3)
logging.info(f"Pytorch tensors with unit values:{t_ones}")

# creating the pytorch tensors with random values
t_random = torch.rand(2, 3)
logging.info(f"Pytorch tensors with random values: {t_random}")

#  manipulating the data type and shape of a tensor
t_a = t_a.to(torch.int64)
logging.info(f"New data type of tensor: {t_a.dtype}")

t_rand = torch.rand(3, 5)
t_rand_trans = torch.transpose(t_rand, 0, 1)
logging.info(f"Transpose of t_rand tesor:{t_rand} is\n {t_rand_trans}")


logging.info(
    f"Removing the unnecessary dimensions (dimensions that have size 1, which are not needed)"
)
t_zeros = torch.zeros(1, 2, 1, 4, 1)
t_sqeeze = torch.squeeze(t_zeros, 2)
logging.info(f"{t_zeros.shape}, ---> , {t_sqeeze.shape}")


logging.info("Multiplying the two random tensors")
t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5, 2))

t3 = torch.mul(t1, t2)
logging.info(f"Multiplying the two random tensors:{t3}")


logging.info("calculating the mean of t1.")
t1_mean = torch.mean(t1, axis=0)
logging.info(f"the mean of t1 is :{t1_mean}")


t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
logging.info(f"Matrix multiplication between t1 and t2 tensor is:{t5}")

# creating the datasetloader for iteration of each element

t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)
for item in data_loader:
    logging.info(item)

# creating the batch of same sizes using the Dataloader class

data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    logging.info(f"Batch {i}: {batch}")


# combining two tensors into a joint dataset

t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)


class JoinDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


joint_dataset = JoinDataset(t_x, t_y)
for example in joint_dataset:
    logging.info("x: %s, y: %s", example[0], example[1])


# Shuffle, batch, and repeat
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
for i, batch in enumerate(data_loader, 1):
    logging.info(f"batch {i}: x: {batch[0]} \n y: {batch[1]}")


# Creating a dataset from files on your local storage disk

image_path = "./"
celeba_dataset = torchvision.datasets.CelebA(
    image_path, split="train", target_type="attr", download=False
)

assert isinstance(celeba_dataset,torch.utils.data.Dataset)
example=next(iter(celeba_dataset))
print(example)
fig = plt.figure(figsize=(12, 8))
for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{attributes[31]}', size=15)
plt.show()