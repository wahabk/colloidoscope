import torch
import numpy as np
from src.trainer import Trainer, LearningRateFinder, predict
from src.unet import UNet
import torchio as tio
from src.dataset import ColloidsDatasetSimulated
from src.deepcolloid import DeepColloid
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
    roiSize = (32,128,128),
    train_data = range(1,39),
    val_data = range(39,51),
    dataset_name = 'replicate',
    batch_size = 2,
    num_workers = 2,
    epochs = 10,
    n_classes = 1,
    lr = 1e-3,
    random_seed = 42,
)
run['parameters'] = params

# hyper-parameters
# params = {
#     "BATCH_SIZE": 8,
#     "DEPTH": 4,
#     "ACTIVATION": "relu",
#     "NORMALIZATION": "group8",
#     "UPSAMPLING": "transposed",
#     "LR": 0.001,
#     "WEIGTH_CE": torch.tensor((0.2, 0.8)),
#     "WEIGTH_DICE": torch.tensor((0.0, 1.0)),
#     "PRECISION": 32,
#     "LR_FINDER": False,
#     "INPUT_SIZE": (128, 128),
#     "CLASSES": 2,
#     "SEED": 42,
#     "EXPERIMENT": "carvana",
#     "MAXEPOCHS": 10,
# }

train_imtrans = tio.Compose([
    # tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 25% of times
    # tio.CropOrPad(roiSize),            # tight crop around brain
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])
train_segtrans = tio.Compose([
    tio.RandomFlip(axes=(0,1,2), flip_probability=0.75)
])

check_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=train_segtrans, label_transform=train_segtrans) 
check_loader = torch.utils.data.DataLoader(check_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
x, y = example = next(iter(check_loader))[0]
print(x.shape, x.max(), x.min())
print(y.shape, y.max(), y.min())


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
                  epochs=params['epochs'],
                  logger=run,
                  )

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

if save:
    model_name =  'output/weights/unet.pt'
    torch.save(model.state_dict(), model_name)
    run['model/weights'].upload(model_name)


# test one predict and upload to neptune
test_array = dc.read_hdf5(params['dataset_name'], 45)
test_label = predict(test_array, 0.3, model, device)
print(test_array.shape, test_label.shape)

array_projection = np.max(test_array, axis=0)
label_projection = np.max(test_label, axis=0)
sidebyside = np.concatenate((array_projection, label_projection), axis=1)
sidebyside /= sidebyside.max()
run['prediction'].upload(File.as_image(sidebyside))

run.stop()
