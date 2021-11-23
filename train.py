import torch
import numpy as np
from src.trainer import Trainer, plot_training, LearningRateFinder
from src.unet import UNet
import torchio as tio
from src.dataset import ColloidsDatasetSimulated
from src.deepcolloid import DeepColloid
import matplotlib.pyplot as plt
import neptune.new as neptune

run = neptune.init(
    project="wahabk/Fishnet",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
)  # your credentials

dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/home/wahab/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)
# dc = DeepColloid(dataset_path)

roiSize = (32,128,128)
train_data = range(1,39)
val_data = range(39,51)
dataset_name = 'replicate'
batch_size = 2
num_workers = 2
epochs=50
n_classes=1
lr = 3e-5
random_seed = 42

train_imtrans = tio.Compose([
    # tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 25% of times
    # tio.CropOrPad(roiSize),            # tight crop around brain
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])
train_segtrans = tio.Compose([
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])

check_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=train_segtrans, label_transform=train_segtrans) 
check_loader = torch.utils.data.DataLoader(check_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
x, y = example = next(iter(check_loader))[0]
print(x.shape, x.max(), x.min())
print(y.shape, y.max(), y.min())


# create a training data loader
train_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=train_segtrans, label_transform=train_segtrans) 
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, val_data) 
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'training on {device}')

# model
model = UNet(in_channels=1,
             out_channels=n_classes,
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
optimizer = torch.optim.Adam(model.parameters(), lr)

# find learning rate
# lrf = LearningRateFinder(model, criterion, optimizer, device)
# lrf.fit(train_loader, steps=1000)
# lrf.plot()

# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_loader,
                  validation_DataLoader=val_loader,
                  lr_scheduler=None,
                  epochs=epochs,
                  )

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

fig = plot_training(training_losses, validation_losses)
plt.show()
plt.savefig('output/loss_curve.png')
