from operator import xor
import torch
from colloidoscope.dataset import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
from colloidoscope.trainer import renormalise
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/home/wahab/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)

train_data = range(1,400)
dataset_name = 'first_run'

train_imtrans = tio.Compose([
    # tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 25% of times
    # tio.CropOrPad(roiSize),            # tight crop around brain
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])
train_segtrans = tio.Compose([
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])

check_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=train_segtrans, label_transform=train_segtrans) 
check_loader = torch.utils.data.DataLoader(check_ds, batch_size=1, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())
x, y = next(iter(check_loader))
# print(example[0].shape)
# print(example[1].shape)

print(x.shape, x.max(), x.min())
print(y.shape, y.max(), y.min())
print(type(x))

x = renormalise(x)
y = renormalise(y)

print(x.shape, x.max(), x.min())
print(y.shape, y.max(), y.min())
print(type(x))

projection = np.max(x, axis=0)
plt.imsave('output/test_genie.png', projection, cmap='gray')
projection = np.max(y, axis=0)
plt.imsave('output/test_genie_label.png', projection, cmap='gray')
exit()