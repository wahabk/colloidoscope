import torch
import numpy as np
from src.trainer import Trainer, LearningRateFinder, predict
from src.unet import UNet
import torchio as tio
from src.dataset import ColloidsDatasetSimulated
from src.deepcolloid import DeepColloid
import matplotlib.pyplot as plt
import neptune.new as neptune
import scipy


dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/home/wahab/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)
# dc = DeepColloid(dataset_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'predict on {device}')

# model
model = UNet(in_channels=1,
			 out_channels=1,
			 n_blocks=4,
			 start_filters=32,
			 activation='relu',
			 normalization='batch',
			 conv_mode='same',
			 dim=3).to(device)

roiSize = (32,128,128)
threshold = 0.3
weights_path =  'output/weights/unet.pt'
dataset_name = 'replicate'

array = dc.read_hdf5(dataset_name, 45)
label, positions = predict(array, threshold, model, device, weights_path, return_positions=True)

array_projection = np.max(array, axis=0)
label_projection = np.max(label, axis=0)
sidebyside = np.concatenate((array_projection, label_projection*255), axis=1)
plt.imsave('output/prediction.png', sidebyside)