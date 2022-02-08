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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os


if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'

	num_samples = 2
	max_num_epochs = 15
	gpus_per_trial = 1
	dataset_name = 'new_year'
	run_name = 'testing'
	save = '/user/home/ak18001/scratch/Colloids/unet.pt'
	device_ids = [0,1]

	config = {
		"lr": tune.loguniform(0.01, 0.001),
		"batch_size": tune.choice([4]),
		"n_blocks": tune.choice([6]),
		"norm": tune.choice(['batch']),
		"epochs": tune.choice([15]),
		"start_filters": tune.choice([32]),
		"activation": tune.choice(['relu']),
	}

	# the scheduler will terminate badly performing trials
	scheduler = ASHAScheduler(
		metric="loss",
		mode="min",
		max_t=max_num_epochs,
		grace_period=1,
		reduction_factor=2)

	result = tune.run(
		partial(train, run_name, dataset_path=dataset_path, dataset_name=dataset_name, train_data=range(1,2000), val_data=range(2001,2501), save=save, tuner=True, device_ids=device_ids),
		resources_per_trial={"cpu": 4, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=scheduler,
		checkpoint_at_end=True)

	best_trial = result.get_best_trial("loss", "min", "last")
	print(f"Best trial config: {best_trial.config}")
	print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

	# TESTING
	# TODO upload to neptune new run

	# setup new neptune run for best run
	run = neptune.init(
		project="wahabk/colloidoscope",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)
	params = dict(
		roiSize = (32,128,128),
		train_data = range(1,2000),
		val_data = range(2001,2501),
		dataset_name = dataset_name,
		batch_size = best_trial.config['batch_size'],
		n_blocks = best_trial.config['n_blocks'],
		norm = best_trial.config['norm'],
		num_workers = 4,
		epochs = best_trial.config['norm'],
		n_classes = 1,
		lr = best_trial.config['lr'],
		random_seed = 42,
		epochs = config['epochs'],
		start_filters = config['start_filters'],
		activation = config['activation'],
	)
	run['Tags'] = 'best config'
	run['parameters'] = params

	best_trained_model = UNet(in_channels=1,
				out_channels=1,
				n_blocks=params['n_blocks'],
				start_filters=32,
				activation='relu',
				normalization=params['norm'],
				conv_mode='same',
				dim=3)

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
		# if gpus_per_trial > 1:
		# 	best_trained_model = torch.nn.DataParallel(best_trained_model)
	best_trained_model.to(device)

	best_checkpoint_dir = best_trial.checkpoint.value
	print(best_checkpoint_dir, 'checkpoint')
	model_state, optimizer_state = torch.load(os.path.join(
		best_checkpoint_dir, "checkpoint"))
	best_trained_model.load_state_dict(model_state)

	train_set = range(3000, 3101)
	losses = test(best_trained_model, dataset_path, dataset_name, train_set, device=device)

	run['test/df'].upload(File.as_html(losses))

	run.stop()