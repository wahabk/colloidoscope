import torch
import numpy as np
from colloidoscope.trainer import Trainer, LearningRateFinder, predict, test
from colloidoscope.unet import UNet
from colloidoscope.dataset import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
import torchio as tio
import matplotlib.pyplot as plt
import neptune.new as neptune
from neptune.new.types import File
import os

print(os.cpu_count())
# fix  cuda multi gpu error
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print('------------num available devices:', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

run = neptune.init(
    project="wahabk/colloidoscope",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
)

# dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
# dataset_path = '/data/mb16907/wahab/Colloids'
dataset_path = '/user/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)

save = True
params = dict(
    roiSize = (32,128,128),
    train_data = range(1,500),
    val_data = range(501,601),
    dataset_name = 'new_year',
    batch_size = 8,
    num_workers = 4,
    epochs = 15,
    n_classes = 1,
    lr = 0.005,
    random_seed = 42,
)
run['Tags'] = 'multigpu 8 blocks'
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
             n_blocks=6,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=3)

model = torch.nn.DataParallel(model, device_ids=[0,1])

model.to(device)

# criterion
criterion = torch.nn.BCEWithLogitsLoss()

# optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), params['lr'])

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
    # run['model/weights'].upload(model_name)

# test one predict and upload to neptune
test_array, metadata, true_positions = dc.read_hdf5(params['dataset_name'], 40, read_metadata=True)
test_label, pred_positions = predict(test_array, model, device, threshold= 0.5, return_positions=True)
print(test_array.shape, test_label.shape)

# TODO add this to test
array_projection = np.max(test_array, axis=0)
label_projection = np.max(test_label, axis=0)*255
sidebyside = np.concatenate((array_projection, label_projection), axis=1)
sidebyside /= sidebyside.max()
run['prediction'].upload(File.as_image(sidebyside))

x, y = dc.get_gr(true_positions, 7, 25)
plt.plot(x, y, label='true')
x, y = dc.get_gr(pred_positions, 7, 25)
plt.plot(x, y, label='pred')
plt.legend()
fig = plt.gcf()
run['gr'].upload(fig)
# plt.savefig('output/gr.png')

test_sample = range(3000, 3101)
losses = test(model, dataset_path, params['dataset_name'], test_sample, device=device, run=run)

run['test/df'].upload(File.as_html(losses))

run.stop()
