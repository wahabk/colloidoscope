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

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)

	dataset_name = 'janpoly'
	test_data = list(range(800,810))

	config = {
		"lr": 0.005,
		"batch_size": 4,
		"n_blocks": 6,
		"norm": 'batch',
		"epochs": 5,
		"start_filters": 32,
		"activation": 'relu',
	}

	params = dict(
		roiSize = (32,128,128),
		dataset_name = dataset_name,
		batch_size = config['batch_size'],
		n_blocks = config['n_blocks'],
		norm = config['norm'],
		num_workers = 4,
		n_classes = 1,
		lr = config['lr'],
		epochs = config['epochs'],
		start_filters = config['start_filters'],
		activation = config['activation'],
		random_seed = 42,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = UNet(in_channels=1,
			out_channels=params['n_classes'],
			n_blocks=params['n_blocks'],
			start_filters=params['start_filters'],
			activation=params['activation'],
			normalization=params['norm'],
			conv_mode='same',
			dim=3)

	model = torch.nn.DataParallel(model)
	model.to(device)

	weights_path = 'output/weights/unet.pt'
	model_weights = torch.load(weights_path, map_location='cpu') # read trained weights
	# print(model_weights.keys())
	model.load_state_dict(model_weights) # add weights to model
	
	losses = test(model, dataset_path, dataset_name, test_data, criterion=torch.nn.L1Loss(), device=device)

	print(losses)