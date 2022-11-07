import torch
import numpy as np
from colloidoscope.deepcolloid import DeepColloid
# from colloidoscope.train_utils import Trainer, LearningRateFinder, predict, test, ColloidsDatasetSimulated
import torchio as tio
from neptune.new.types import File
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
from ray import tune
import random
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from pathlib2 import Path

import copy
import monai
import math
from monai.networks.layers.factories import Act, Norm

print(os.cpu_count())
print(torch.cuda.is_available())
print ('Current cuda device ', torch.cuda.current_device())
print('------------num available devices:', torch.cuda.device_count())

import torch.nn as nn
import torch.nn.functional as F 

from b01 import train

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)

	dataset_name = 'new_1400_30nm'
	n_samples = dc.get_hdf5_keys(dataset_name)
	print(len(n_samples))
	all_data = list(range(1,1400))
	random.shuffle(all_data)

	train_data = all_data[0:1000]
	val_data = all_data[1000:1200]
	test_data =	list(range(1,498))
	# train_data = all_data[0:200]
	# val_data = all_data[200:250]
	# test_data =	list(range(1,50))
	name = 'search LF lr'
	# save = '/home/ak18001/code/colloidoscope/output/weights/unet.pt'
	device_ids = [0,]
	save = False
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	num_samples = 40
	max_num_epochs = 45
	gpus_per_trial = 1

	config = {
		"lr": tune.loguniform(0.01, 0.00001),
		"batch_size": 8,
		"n_blocks": 3,
		"norm": "BATCH",
		"epochs": 40,
		"start_filters": 32,
		"activation": "RELU",
		"dropout": 0.2,
		"loss_function": torch.nn.L1Loss()#tune.choice([torch.nn.L1Loss()]) #BinaryFocalLoss(alpha=1.5, gamma=0.5), 
	}

	# the scheduler will terminate badly performing trials
	# scheduler = ASHAScheduler(
	# 	metric="val_loss",
	# 	mode="min",
	# 	max_t=max_num_epochs,
	# 	grace_period=1,
	# 	reduction_factor=2)

	# print(f"LOCAL PATH {Path().parent.resolve()} \n\n")
	work_dir = Path().parent.resolve()

	result = tune.run(
		partial(train, name=name, dataset_path=dataset_path, dataset_name=dataset_name, 
				train_data=train_data, val_data=val_data, test_data=test_data, save=save, 
				tuner=True, device_ids=device_ids, work_dir=work_dir),
		resources_per_trial={"cpu": 10, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=None,
		checkpoint_at_end=False,
		local_dir='/home/ak18001/Data/HDD/Colloids/ray_results/') # Path().parent.resolve()/'ray_results'
