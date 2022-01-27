from operator import xor
from cv2 import flip
import torch
from colloidoscope.dataset import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
from colloidoscope.trainer import renormalise
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # dataset_path = '/home/ak18001/Data/HDD/Colloids'
    dataset_path = '/home/wahab/Data/HDD/Colloids'
    # dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
    dc = DeepColloid(dataset_path)

    

    train_data = range(1,5000)
    dataset_name = 'new_year'


    transforms_affine = tio.Compose([
        tio.RandomFlip(axes=(1,2), flip_probability=1)
    ])
    transforms_img = tio.Compose([
        tio.RandomAnisotropy(p=1),              # make images look anisotropic 25% of times
    ])


    check_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=transforms_img, label_transform=transforms_affine) 
    check_loader = torch.utils.data.DataLoader(check_ds, batch_size=1, shuffle=True, num_workers=1)
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

    dc.view(x, label=y)

    array_projection = np.max(x, axis=0)
    label_projection = np.max(y, axis=0)*255
    sidebyside = np.concatenate((array_projection, label_projection), axis=1)
    sidebyside /= sidebyside.max()
    plt.imsave('output/test_genie.png', sidebyside, cmap='gray')

    