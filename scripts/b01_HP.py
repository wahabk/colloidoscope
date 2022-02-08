import torch
import numpy as np
from colloidoscope.trainer import train
from colloidoscope.deepcolloid import DeepColloid
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
from ray import tune
import random
from ray.tune.schedulers import ASHAScheduler
from functools import partial

print(os.cpu_count())
print(torch.cuda.is_available())
print ('Current cuda device ', torch.cuda.current_device())
print('------------num available devices:', torch.cuda.device_count())

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)


	dataset_name = 'feb_blur'
	n_samples = dc.get_hdf5_keys(dataset_name)
	print(len(n_samples))
	all_data = list(range(2,990))
	random.shuffle(all_data)

	num_samples = 10
	max_num_epochs = 10
	gpus_per_trial = 1

	train_data = all_data[0:600]
	val_data = all_data[601:801]
	test_data =	all_data[801:]
	name = 'test'
	save = 'output/weights/unet.pt'
	device_ids = [0,]
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	config = {
		"lr": tune.loguniform(0.01, 0.001),
		"epochs": tune.choice([10]),
		"batch_size": tune.choice([4]),
		"n_blocks": tune.choice([6]),
		"norm": tune.choice(['batch']),
		"activation": tune.choice(['relu'], 'leaky_relu'),
		"start_filters": tune.randint(3,33),
		"loss": tune.choice([torch.nn.CrossEntropyLoss(), torch.nn.L1Loss(), torch.nn.MSELoss()]),
	}

		# the scheduler will terminate badly performing trials
	scheduler = ASHAScheduler(
		metric="loss",
		mode="min",
		max_t=max_num_epochs,
		grace_period=1,
		reduction_factor=2)

	result = tune.run(
		partial(train, name, dataset_path=dataset_path, dataset_name=dataset_name, 
				train_data=range(1,2000), val_data=range(2001,2501), save=save, 
				tuner=True, device_ids=device_ids),
		resources_per_trial={"cpu": 4, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=scheduler,
		checkpoint_at_end=True)