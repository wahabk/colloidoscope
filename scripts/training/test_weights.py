import torch
import numpy as np
from colloidoscope.train_utils import Trainer, LearningRateFinder, test, ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
import torchio as tio
from neptune.new.types import File
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
from ray import tune
import random
import copy
import monai
import math
from pathlib2 import Path
from monai.networks.layers.factories import Act, Norm


if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)

	dataset_name = 'heatmap_1400'
	num_workers=16
	test_data =	list(range(1,599))
	random.shuffle(test_data)
	test_data = test_data[:20]
	name="TW: first test weights"

	post_processing = 'tp'

	config = {
		"lr": 0.002165988,
		"batch_size": 16,
		"n_blocks": 2,
		"norm": 'INSTANCE',
		"epochs": 6,
		"start_filters": 32,
		"activation": "SWISH",
		"dropout": 0.1,
		"loss_function": torch.nn.L1Loss(), #torch.nn.BCEWithLogitsLoss() #BinaryFocalLoss(alpha=1.5, gamma=0.5),
	}

	params = dict(
		roiSize = (100,100,100),
		label_size = (96,96,96),
		# train_data = train_data,
		# val_data = val_data,
		test_data = test_data,
		dataset_name = dataset_name,
		batch_size = config['batch_size'],
		n_blocks = config['n_blocks'],
		norm = config['norm'],
		loss_function = config['loss_function'],
		lr = config['lr'],
		epochs = config['epochs'],
		start_filters = config['start_filters'],
		activation = config['activation'],
		dropout = config['dropout'],
		num_workers = num_workers,
		n_classes = 1,
		random_seed = 42,
	)
	

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	start_filters = params['start_filters']
	n_blocks = params['n_blocks']
	start = int(math.sqrt(start_filters))
	channels = [2**n for n in range(start, start + n_blocks)]
	strides = [2 for n in range(1, n_blocks)]
	model = monai.networks.nets.AttentionUnet(
		spatial_dims=3,
		in_channels=1,
		out_channels=params['n_classes'],
		channels=channels,
		strides=strides,
		kernel_size=7,
		# up_kernel_size=3,
		dropout=params["dropout"],
		padding='valid',
	)

	model = torch.nn.DataParallel(model)
	model.to(device)

	weights_path = 'output/weights/attention_unet_202211.pt'
	model_weights = torch.load(weights_path, map_location='cpu') # read trained weights
	# print(model_weights.keys())
	model.load_state_dict(model_weights) # add weights to model

	run = neptune.init_run(
		project="wahabk/colloidoscope",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)
	run['Tags'] = str(name)
	run['parameters'] = params

	work_dir = Path().parent.resolve()
	
	losses = test(model, dataset_path, dataset_name, test_data, canvas_size=params['roiSize'], label_size=params['label_size'],
				post_processing=post_processing, criterion=torch.nn.L1Loss(), device=device, work_dir=work_dir, run=run)

	run['test/df'].upload(File.as_html(losses))
	run.stop()