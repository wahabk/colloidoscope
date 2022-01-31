import torch
import numpy as np
from colloidoscope.trainer import Trainer, LearningRateFinder, predict, test, train
from colloidoscope.unet import UNet
from colloidoscope.dataset import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
import torchio as tio
import matplotlib.pyplot as plt
import neptune.new as neptune
from neptune.new.types import File
import os
from ray import tune

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())

# fix  cuda multi gpu error
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print('------------num available devices:', torch.cuda.device_count())

if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	dataset_path = '/user/home/ak18001/scratch/Colloids/'
	dc = DeepColloid(dataset_path)

	dataset_name = 'new_year'
	train_data=range(2,2000)
	val_data=range(2001,2501)
	name='testing'
	# save = 'output/weights/unet.pt'
	save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	config = {
		"lr": 0.005,
		"batch_size": 8,
		"n_blocks": 6,
		"norm": 'batch',
		"epochs": 30,
		"start_filters": 32,
		"activation": 'relu',
	}

	train(config, name, dataset_path=dataset_path, dataset_name=dataset_name, 
				train_data=train_data, val_data=val_data, save=save, tuner=False)
