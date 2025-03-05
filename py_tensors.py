import torch
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')

# creating the pytorch tensors
a=[1,2,3]
t_a=torch.tensor(a)
logging.info(f'Pytorch tensors: {t_a}')

# creating the pytorch tensors with values 1
t_ones=torch.ones(2,3)
logging.info(f'Pytorch tensors with unit values:{t_ones}')

# creating the pytorch tensors with random values
t_random=torch.rand(2,3)
logging.info(f'Pytorch tensors with random values: {t_random}')

#  manipulating the data type and shape of a tensor
t_a=t_a.to(torch.int64)
logging.info(f'New data type of tensor: {t_a.dtype}')

t_rand=torch.rand(3,5)
t_rand_trans=torch.transpose(t_rand,0,1)
logging.info(f'Transpose of t_rand tesor:{t_rand} is\n {t_rand_trans}')


logging.info(f'Removing the unnecessary dimensions (dimensions that have size 1, which are not needed)')
t_zeros= torch.zeros(1, 2, 1, 4, 1)
t_sqeeze=torch.squeeze(t_zeros,2)
logging.info(f'{t_zeros.shape}, ---> , {t_sqeeze.shape}')