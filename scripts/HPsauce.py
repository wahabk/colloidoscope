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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os

def train(config, dataset_path, dataset_name='new_year'):
	dc = DeepColloid(dataset_path)

	# setup neptune
	run = neptune.init(
		project="wahabk/colloidoscope",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)
	params = dict(
		roiSize = (32,128,128),
		train_data = range(1,2000),
		val_data = range(2001,2501),
		dataset_name = dataset_name,
		batch_size = config['batch_size'],
		n_blocks = config['n_blocks'],
		norm = config['norm'],
		num_workers = 4,
		epochs = 2,
		n_classes = 1,
		lr = config['lr'],
		random_seed = 42,
	)
	run['Tags'] = 'trying hpsauce test'
	run['parameters'] = params

	train_imtrans = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.75),
		# tio.RandomElasticDeformation(num_control_points=7,max_displacement=2),
		# tio.RandomAnisotropy((0,1,2), p=0.1),
		# tio.RandomBiasField(0.5),
	])
	train_segtrans = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.75),
		# tio.RandomElasticDeformation(num_control_points=7,max_displacement=2),
	])

	# create a training data loader
	train_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=train_imtrans, label_transform=train_segtrans) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
	# create a validation data loader
	val_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['val_data']) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())

	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	# model
	#TODO add model params to neptune
	model = UNet(in_channels=1,
				out_channels=params['n_classes'],
				n_blocks=params['n_blocks'],
				start_filters=32,
				activation='relu',
				normalization=params['norm'],
				conv_mode='same',
				dim=3).to(device)

	# loss function
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
					tuner=True,
					)

	# start training
	training_losses, validation_losses, lr_rates = trainer.run_trainer()

	# test one predict and upload to neptune
	test_array, metadata, positions = dc.read_hdf5(params['dataset_name'], 45)
	test_label = predict(test_array, model, device, threshold=0.5, return_positions=False)
	array_projection = np.max(test_array, axis=0)
	label_projection = np.max(test_label, axis=0)*255
	sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	sidebyside /= sidebyside.max()
	run['prediction'].upload(File.as_image(sidebyside))

	run.stop()




if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'

	num_samples = 2
	max_num_epochs = 15 # TODO feed this to train()
	gpus_per_trial = 1
	dataset_name = 'new_year'

	config = {
	# "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
	# "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
	"lr": tune.loguniform(0.01, 0.001),
	"batch_size": tune.choice([4]),
	"n_blocks": tune.choice([6]),
	"norm": tune.choice(['batch'])
	}

	# the scheduler will terminate badly performing trials
	scheduler = ASHAScheduler(
		metric="loss",
		mode="min",
		max_t=max_num_epochs,
		grace_period=1,
		reduction_factor=2)

	reporter = CLIReporter(
		# parameter_columns=["l1", "l2", "lr", "batch_size"],
		metric_columns=["loss", "training_iteration"])


	result = tune.run(
		partial(train, dataset_path=dataset_path, dataset_name=dataset_name),
		resources_per_trial={"cpu": 4, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=scheduler,
		# progress_reporter=reporter,
		checkpoint_at_end=True)

	best_trial = result.get_best_trial("loss", "min", "last")
	print(f"Best trial config: {best_trial.config}")
	print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

	# TESTING
	# TODO upload to neptune new run

	# setup neptune
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
		epochs = 2,
		n_classes = 1,
		lr = best_trial.config['lr'],
		random_seed = 42,
	)
	run['Tags'] = 'testing best config'
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