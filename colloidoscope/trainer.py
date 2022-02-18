import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import math
import neptune.new as neptune
from neptune.new.types import File
from .deepcolloid import *
from .dataset import *
from .unet import UNet
from .unet_dsnt import UNetCC
from ray import tune
import os
import torchio as tio
import scipy
from .predict import predict
from torch.nn import BCELoss
import copy

class Trainer:
	def __init__(self,
				 model: torch.nn.Module,
				 device: torch.device,
				 criterion: torch.nn.Module,
				 optimizer: torch.optim.Optimizer,
				 training_DataLoader: torch.utils.data.Dataset,
				 validation_DataLoader: torch.utils.data.Dataset = None,
				 lr_scheduler: torch.optim.lr_scheduler = None,
				 epochs: int = 100,
				 epoch: int = 0,
				 notebook: bool = False,
				 logger=None,
				 tuner=False,
				 ):

		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.training_DataLoader = training_DataLoader
		self.validation_DataLoader = validation_DataLoader
		self.device = device
		self.epochs = epochs
		self.epoch = epoch
		self.notebook = notebook
		self.logger = logger
		self.tuner = tuner

		self.training_loss = []
		self.validation_loss = []
		self.learning_rate = []

	def run_trainer(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		progressbar = trange(self.epochs, desc='Progress')
		for i in progressbar:
			"""Epoch counter"""
			self.epoch += 1  # epoch counter

			"""Training block"""
			self._train()

			if self.logger: self.logger['epochs/loss'].log(self.training_loss[-1])

			"""Validation block"""
			if self.validation_DataLoader is not None:
				self._validate()

			if self.logger: self.logger['epochs/val'].log(self.validation_loss[-1])

			if self.tuner:
				with tune.checkpoint_dir(self.epoch) as checkpoint_dir:
					path = os.path.join(checkpoint_dir, "checkpoint")
					torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)


			"""Learning rate scheduler block"""
			if self.lr_scheduler is not None:
				if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
					self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
				else:
					self.lr_scheduler.batch()  # learning rate scheduler step
		return self.training_loss, self.validation_loss, self.learning_rate

	def _train(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		self.model.train()  # train mode
		train_losses = []  # accumulate the losses here
		batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
						  leave=False)

		for i, (x, y) in batch_iter:
			input_, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
			self.optimizer.zero_grad()  # zerograd the parameters
			out = self.model(input_)  # one forward pass
			# if self.criterion == 'BCELoss()':
			# 	out_sigmoid = torch.nn.Sigmoid(out)
			# 	loss_value = self.criterion(out_sigmoid, target)  # calculate loss

			loss = self.criterion(out, target)  # calculate loss
			loss_value = loss.item()
			train_losses.append(loss_value)
			if self.logger: self.logger['train/loss'].log(loss_value)
			loss.backward()  # one backward pass
			self.optimizer.step()  # update the parameters

			batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

		self.training_loss.append(np.mean(train_losses))
		self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

		batch_iter.close()

	def _validate(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		self.model.eval()  # evaluation mode
		valid_losses = []  # accumulate the losses here
		batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
						  leave=False)

		for i, (x, y) in batch_iter:
			input_, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

			with torch.no_grad():
				out = self.model(input_)
				loss = self.criterion(out, target)
				loss_value = loss.item()
				valid_losses.append(loss_value)
				if self.logger: self.logger['val/loss'].log(loss_value)
				batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

		self.validation_loss.append(np.mean(valid_losses))
		if self.tuner: tune.report(val_loss=(np.mean(valid_losses)))


		batch_iter.close()

class LearningRateFinder:
	"""
	Train a model using different learning rates within a range to find the optimal learning rate.
	"""

	def __init__(self,
				 model: torch.nn.Module,
				 criterion,
				 optimizer,
				 device
				 ):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.loss_history = {}
		self._model_init = model.state_dict()
		self._opt_init = optimizer.state_dict()
		self.device = device

	def fit(self,
			data_loader: torch.utils.data.DataLoader,
			steps=100,
			min_lr=1e-7,
			max_lr=1,
			constant_increment=False,
			):
		"""
		Trains the model for number of steps using varied learning rate and store the statistics
		"""
		self.loss_history = {}
		self.model.train()
		current_lr = min_lr
		steps_counter = 0
		epochs = math.ceil(steps / len(data_loader))

		progressbar = trange(epochs, desc='Progress')
		for epoch in progressbar:
			batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
							  leave=False)

			for i, (x, y) in batch_iter:
				x, y = x.to(self.device), y.to(self.device)
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = current_lr
				self.optimizer.zero_grad()
				out = self.model(x)
				loss = self.criterion(out, y)
				loss.backward()
				self.optimizer.step()
				self.loss_history[current_lr] = loss.item()

				steps_counter += 1
				if steps_counter > steps:
					break

				if constant_increment:
					current_lr += (max_lr - min_lr) / steps
				else:
					current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)


	def plot(self,
			 smoothing=True,
			 clipping=True,
			 smoothing_factor=0.1
			 ):
		"""
		Shows loss vs learning rate(log scale) in a matplotlib plot
		"""
		loss_data = pd.Series(list(self.loss_history.values()))
		lr_list = list(self.loss_history.keys())
		if smoothing:
			loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
			loss_data = loss_data.divide(pd.Series(
				[1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]))  # bias correction
		if clipping:
			loss_data = loss_data[10:-5]
			lr_list = lr_list[10:-5]
		plt.figure()
		plt.plot(lr_list, loss_data)
		plt.xscale('log')
		plt.title('Loss vs Learning rate')
		plt.xlabel('Learning rate (log scale)')
		plt.ylabel('Loss (exponential moving average)')
		plt.savefig('output/learning_rate_finder.png')

	def reset(self):
		"""
		Resets the model and optimizer to its initial state
		"""
		self.model.load_state_dict(self._model_init)
		self.optimizer.load_state_dict(self._opt_init)
		print('Model and optimizer in initial state.')


def dep_test(model, train_data, device='cpu'):

	for d in train_data:
		label, positions = predict(d, model, device=device, return_positions=True)

def find_positions(result, threshold) -> np.ndarray:
	label = result.copy()

	label[label > threshold] = 1
	label[label < threshold] = 0

	str_3D=np.array([[[0, 0, 0],
					[0, 1, 0],
					[0, 0, 0]],

					[[0, 1, 0],
					[1, 1, 1],
					[0, 1, 0]],

					[[0, 0, 0],
					[0, 1, 0],
					[0, 0, 0]]], dtype='uint8')

	resultLabel = scipy.ndimage.label(label, structure=str_3D)
	positions = scipy.ndimage.center_of_mass(result, resultLabel[0], index=range(1,resultLabel[1]))
	return np.array(positions)

def test(model, dataset_path, dataset_name, test_set, threshold=0.5, num_workers=4, batch_size=1, criterion=torch.nn.BCEWithLogitsLoss(), run=False, device='cpu'):
	dc = DeepColloid(dataset_path)

	print('Running test, this may take a while...')
	
	test_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, test_set, return_metadata=False) 
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

	losses = {}
	first = True # save figs for first instance
	model.eval()
	with torch.no_grad():
		for idx, data in enumerate(test_loader):
			i = test_set[idx]
			metadata, true_positions, diameters = dc.read_metadata(dataset_name, i)
			#fix for now
			diameters = [metadata['params']['r']*2 for i in range(len(true_positions))]

			x, y = data
			x, y = x.to(device), y.to(device)
			# print(x.shape, x.max(), x.min())
			# print(y.shape, y.max(), y.min())

			out = model(x)  # send through model/network
			loss = criterion(out, y)
			loss = loss.cpu().numpy()
			out_relu = torch.relu(out)
			out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits
			# post process to numpy array
			result = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray
			result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

			pred_positions = find_positions(result, threshold)
			prec, rec = dc.get_precision_recall(true_positions, pred_positions, diameters, 0.5,)
			print(pred_positions.shape)

			m = {
				'dataset': metadata['dataset'],
				'n' 	 : metadata['n'],
				'idx'	 : i,
				'volfrac': metadata['volfrac'],
				'n_particles':  metadata['n_particles'],
				**metadata['params'],
				'loss'	 : float(loss),
				'precision'	 : float(prec),
				'recall'	 : float(rec),
			}

			if first and run:
				if len(pred_positions) == 0:
					print('Skipping gr() as bad pred')
				else:		
					x, y = dc.get_gr(true_positions, 50, 50)
					plt.plot(x, y, label='true')
					x, y = dc.get_gr(pred_positions, 50, 50)
					plt.plot(x, y, label='pred')
					plt.legend()
					fig = plt.gcf()
					run['gr'].upload(fig)
					plt.clf()

				ap, precisions, recalls, thresholds = dc.average_precision(true_positions, pred_positions, diameters=diameters)
				run['AP'] = ap

				fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='Unet')
				run['PR_curve'].upload(fig)
				plt.clf()

				first = False
			losses[idx] = m

	return losses

def renormalise(tensor: torch.Tensor):
	array = tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	array = np.squeeze(array)  # remove batch dim and channel dim -> [H, W]
	array = array * 255
	return array

def train(config, name, dataset_path, dataset_name, train_data, val_data, test_data, save=False, tuner=True, device_ids=[0,1], num_workers=10):
	'''
	by default for ray tune
	'''

	dc = DeepColloid(dataset_path)

	# setup neptune
	run = neptune.init(
		project="wahabk/colloidoscope",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)
	params = dict(
		roiSize = (64,64,64),
		train_data = train_data,
		val_data = val_data,
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
		num_workers = num_workers,
		n_classes = 1,
		random_seed = 42,
	)
	run['Tags'] = name
	run['parameters'] = params

	transforms_affine = tio.Compose([
		tio.RandomFlip(axes=(1,2), flip_probability=0.5),
		# tio.RandomAffine(),
	])
	transforms_img = tio.Compose([
		tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 25% of times
		tio.RandomBlur(p=0.1),
		# tio.OneOf({
		# 	tio.RandomNoise(0.1, 0.01): 0.1,
		# 	tio.RandomBiasField(0.1): 0.1,
		# 	tio.RandomGamma((-0.3,0.3)): 0.1,
		# 	tio.RandomMotion(): 0.3,
		# }),
		tio.RescaleIntensity((0.05,0.95)),
	])

	# create a training data loader
	train_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=transforms_img, label_transform=transforms_affine) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
	# create a validation data loader
	val_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['val_data']) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())

	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	# model
	model = UNetCC(in_channels=1,
				out_channels=params['n_classes'],
				n_blocks=params['n_blocks'],
				start_filters=params['start_filters'],
				activation=params['activation'],
				normalization=params['norm'],
				conv_mode='same',
				dim=3,
				skip_connect=None)

	model = torch.nn.DataParallel(model, device_ids=device_ids)
	model.to(device)

	# loss function
	# criterion = torch.nn.BCEWithLogitsLoss()
	criterion = params['loss_function']

	params['loss_function'] = str(copy.deepcopy(params['loss_function']))

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
					tuner=tuner,
					)

	# start training
	training_losses, validation_losses, lr_rates = trainer.run_trainer()
	
	if save:
		model_name = save
		torch.save(model.state_dict(), model_name)
		# run['model/weights'].upload(model_name)

	# test one predict and upload to neptune
	data = dc.read_hdf5(params['dataset_name'], 1)
	test_array, metadata, positions = data['image'], data['metadata'], data['positions']
	test_label = predict(test_array, model, device, threshold=0.5, return_positions=False)
	array_projection = np.max(test_array, axis=0)
	label_projection = np.max(test_label, axis=0)*255
	sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	sidebyside /= sidebyside.max()
	run['prediction'].upload(File.as_image(sidebyside))

	losses = test(model, dataset_path, dataset_name, test_data, run=run, criterion=criterion, device=device, num_workers=num_workers)
	
	losses = pd.DataFrame(losses)
	run['test/df'].upload(File.as_html(losses))
	# run['test/test'].log(losses) #if dict

	run.stop()
