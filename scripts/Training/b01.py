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
import random

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)

	dataset_name = 'feb_psf_const'
	n_samples = dc.get_hdf5_keys(dataset_name)
	print(len(n_samples))
	all_data = list(range(2,990))
	random.shuffle(all_data)

	train_data = all_data[0:600]
	val_data = all_data[601:801]
	test_data =	all_data[801:901]
	name = 'new_augs'
	save = 'output/weights/unet.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	config = {
		"lr": 0.00112668,
		"batch_size": 4,
		"n_blocks": 5,
		"norm": 'batch',
		"epochs": 2,
		"start_filters": 32,
		"activation": 'relu',
		"loss_function": torch.nn.L1Loss(),
	}

	train(config, name, dataset_path=dataset_path, dataset_name=dataset_name, 
				train_data=train_data, val_data=val_data, test_data=test_data, 
				save=save, tuner=False, device_ids=[0,])
