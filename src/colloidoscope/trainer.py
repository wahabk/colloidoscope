import numpy as np
import torch
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import math
import neptune.new as neptune
from .deepcolloid import DeepColloid
import scipy

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

			"""Validation block"""
			if self.validation_DataLoader is not None:
				self._validate()

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
			input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
			self.optimizer.zero_grad()  # zerograd the parameters
			out = self.model(input)  # one forward pass
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
			input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

			with torch.no_grad():
				out = self.model(input)
				loss = self.criterion(out, target)
				loss_value = loss.item()
				valid_losses.append(loss_value)
				if self.logger: self.logger['val/loss'].log(loss_value)
				batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

		self.validation_loss.append(np.mean(valid_losses))

		batch_iter.close()

class LearningRateFinder:
	"""
	Train a model using different learning rates within a range to find the optimal learning rate.
	"""

	def __init__(self,
				 model: nn.Module,
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
			constant_increment=False
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


def predict(scan, threshold, model, device, weights_path=None, return_positions=False):
	
	if weights_path:
		model_weights = torch.load(weights_path) # read trained weights
		model.load_state_dict(model_weights) # add weights to model

	array = scan.copy()
	array = np.array(array/255, dtype=np.float32)
	array = np.expand_dims(array, 0)      # add channel axis
	array = np.expand_dims(array, 0)      # add batch axis
	array = torch.from_numpy(array).to(device)  # to torch, send to device
	
	model.eval()
	with torch.no_grad():
		out = model(array)  # send through model/network

	out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output
	
	# post process to numpy array
	result = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]
	label = result.copy()
	label[label > threshold] = 1
	label[label < threshold] = 0

	if return_positions:
		str_3D=np.array([
		[[0, 0, 0],
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
		positions = np.array(positions)
		return result, positions
	else:
		return result

