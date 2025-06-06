import torch
import numpy as np
from colloidoscope.trainer import Trainer, LearningRateFinder, predict
from colloidoscope.unet import UNet
from colloidoscope.dataset import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
import torchio as tio
import matplotlib.pyplot as plt
import neptune.new as neptune
from neptune.new.types import File

run = neptune.init(
    project="wahabk/colloidoscope",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
)

dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/home/wahab/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)


save = True
params = dict(
    tag = 'finding lr',
    roiSize = (32,128,128),
    train_data = range(1,1000),
    val_data = range(1001,1500),
    dataset_name = 'new_year',
    batch_size = 4,
    num_workers = 4,
    epochs = 15,
    n_classes = 1,
    lr = 10e-4,
    random_seed = 42,
)
run['Tags'] = 'NY finding LR'
run['parameters'] = params

train_imtrans = tio.Compose([
    # tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 25% of times
    # tio.CropOrPad(roiSize),            # tight crop around brain
    # tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])
train_segtrans = tio.Compose([
    # tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])

check_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=train_segtrans, label_transform=train_segtrans) 
check_loader = torch.utils.data.DataLoader(check_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
for i in check_loader:
    batch = i
    break
x,y = batch
print(x[0].shape, x[0].max(), x[0].min())
print(y[0].shape, y[0].max(), y[0].min())


# create a training data loader
train_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=train_segtrans, label_transform=train_segtrans) 
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['val_data']) 
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'training on {device}')

# model
#TODO add model params to neptune
model = UNet(in_channels=1,
             out_channels=params['n_classes'],
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3).to(device)

# criterion
criterion = torch.nn.BCEWithLogitsLoss()

# optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), params['lr'])

# find learning rate
lrf = LearningRateFinder(model, criterion, optimizer, device)
lrf.fit(train_loader, steps=500)
lrf.plot()

run.stop()