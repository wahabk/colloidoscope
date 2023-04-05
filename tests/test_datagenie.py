from operator import xor
from cv2 import flip
import torch
from colloidoscope.train_utils import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
# from colloidoscope.trainer import renormalise
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # dataset_path = '/home/ak18001/Data/HDD/Colloids'
    # dataset_path = '/home/wahab/Data/HDD/Colloids'
    dataset_path = '/mnt/scratch/ak18001/Colloids'
    dc = DeepColloid(dataset_path)

    

    train_data = [1]
    dataset_name = 'heatmap_3000_2'


    transforms_affine = tio.Compose([
        tio.RandomFlip(axes=(0,1,2), flip_probability=1),
        tio.RandomAffine(),
    ])
    transforms_img = tio.Compose([
        tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
        tio.RandomBlur(p=0.25),
        tio.OneOf({
            tio.RandomNoise(0.25, 0.01): 0.25,
            tio.RandomBiasField(0.1): 0.25,
            tio.RandomGamma((-0.3,0.3)): 0.5,
            tio.RandomMotion(): 0.3,
        }),
        tio.RescaleIntensity((0.05,0.95)),
    ])


    check_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=transforms_img, label_transform=transforms_affine) 
    check_loader = torch.utils.data.DataLoader(check_ds, batch_size=4, shuffle=True, num_workers=1)
    # x, y = next(iter(check_loader))
    # # print(example[0].shape)
    # # print(example[1].shape)

    # print(x.shape, x.max(), x.min())
    # print(y.shape, y.max(), y.min())
    # print(type(x))

    # x = renormalise(x)
    # y = renormalise(y)

    # print(x.shape, x.max(), x.min())
    # print(y.shape, y.max(), y.min())
    # print(type(x))

    # dc.view(x, label=y)

    # array_projection = np.max(x, axis=0)
    # label_projection = np.max(y, axis=0)
    # sidebyside = np.concatenate((array_projection, label_projection), axis=1)
    # sidebyside /= sidebyside.max()
    # plt.imsave('output/test_genie.png', sidebyside, cmap='gray')

    d = dc.read_hdf5(dataset_name, n=train_data[0])
    x = d['image']

    im_projs = [x.max(axis=0), x.max(axis=1)]
    im_proj = np.concatenate(im_projs, axis=0)

    plt.imsave(f"output/figs/data_aug/data_aug_raw.png", im_proj)

    # import pdb;pdb.set_trace()
    saved = 0
    target = 5
    for e in range(target):
        for batch in check_loader:
            xs, ys = batch
            xs, ys = xs.cpu(), ys.cpu()

            for x,y in zip(xs, ys):
                print(x.shape, y.shape)

                x = np.squeeze(np.array(x.numpy()))*255
                # y = undo_one_hot(np.array(y), n_classes=5)
                y = np.squeeze(np.array(y.numpy()))*255
                print(x.max(), x.min())
                
                im_projs = [x.max(axis=0), x.max(axis=1)]
                im_proj = np.concatenate(im_projs, axis=0)
                
                plt.imsave(f"output/figs/data_aug/data_aug_{saved}.png", im_proj)
                saved+=1
                if saved >= target: exit()

    