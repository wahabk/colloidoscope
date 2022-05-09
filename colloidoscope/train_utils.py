"""
Colloidoscope train_utils

This file contains:
- Pytorch datasets
- Pytorch Trainer
- training and testing functions
- training utilities

"""

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import math
import os
from skimage import io
import seaborn as sns
import copy

import torch
import neptune.new as neptune
from neptune.new.types import File
from ray import tune
import torchio as tio

from .deepcolloid import DeepColloid
from .models.unet import UNet
from .simulator import crop_positions_for_label
from .predict import *
from torch.nn import BCELoss
import torch.nn.functional as F

"""
Datasets
"""

class ColloidsDatasetSimulated(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for simulated colloids

	transform is augmentation function

	"""	

	def __init__(self, dataset_path:str, dataset_name:str, indices:list, transform=None, label_transform=None, return_metadata=False, label_size:tuple=(64,64,64)):	
		super().__init__()
		self.dataset_path = dataset_path
		self.dataset_name = dataset_name
		self.indices = indices
		self.transform = transform
		self.label_transform = label_transform
		self.return_metadata = return_metadata
		self.label_size = label_size


	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		dc = DeepColloid(self.dataset_path)
		# Select sample
		i = self.indices[index]

		data = dc.read_hdf5(self.dataset_name, i)
		X, y, positions, metadata = data['image'], data['label'], data['positions'], data['metadata']

		# dc.view(X)
		# napari.run()

		y = dc.crop3d(y, self.label_size)

		X = np.array(X/X.max(), dtype=np.float32)
		y = np.array(y/y.max() , dtype=np.float32)
		
		# print('x', np.min(X), np.max(X), X.shape)
		# print('y', np.min(y), np.max(y), y.shape)

		#for reshaping
		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor

		# import pdb; pdb.set_trace()

		X = torch.from_numpy(X)
		y = torch.from_numpy(y)

		if self.transform:
			if self.label_transform:
				stacked = torch.cat([X, y], dim=0) # shape=(2xHxW)
				stacked = self.label_transform(stacked)
				X, y = torch.chunk(stacked, chunks=2, dim=0)
			X = self.transform(X)

		# print('x', np.min(X), np.max(X), X.shape)
		# print('y', np.min(y), np.max(y), y.shape)

		return X, y,


def compute_max_depth(shape=1920, max_depth=10, print_out=True):
    shapes = []
    shapes.append(shape)
    for level in range(1, max_depth):
        if shape % 2 ** level == 0 and shape / 2 ** level > 1:
            shapes.append(shape / 2 ** level)
            if print_out:
                print(f'Level {level}: {shape / 2 ** level}')
        else:
            if print_out:
                print(f'Max-level: {level - 1}')
            break
    return shapes

"""
Train and test
"""


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
	#TODO find a way to precalculate this - should i only unpad the first block?
	if config['n_blocks'] == 2: label_size = (48,48,48)
	if config['n_blocks'] == 3: label_size = (24,24,24)

	transforms_affine = tio.Compose([
		# tio.RandomFlip(axes=(1,2), flip_probability=0.5),
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
	train_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=transforms_img, label_transform=None, label_size=label_size) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
	# create a validation data loader
	val_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['val_data'], label_size=label_size) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())

	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	# model
	model = UNet(in_channels=1,
				out_channels=params['n_classes'],
				n_blocks=params['n_blocks'],
				start_filters=params['start_filters'],
				activation=params['activation'],
				normalization=params['norm'],
				conv_mode='valid',
				up_mode='transposed',
				dim=3,
				skip_connect=None,
				)

	model = torch.nn.DataParallel(model, device_ids=device_ids)
	model.to(device)

	# loss function
	# criterion = torch.nn.BCEWithLogitsLoss()
	criterion = params['loss_function']

	params['loss_function'] = str(copy.deepcopy(params['loss_function']))

	# optimizer
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	optimizer = torch.optim.Adam(model.parameters(), params['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
	# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, cycle_momentum=False)

	# trainer
	trainer = Trainer(model=model,
					device=device,
					criterion=criterion,
					optimizer=optimizer,
					training_DataLoader=train_loader,
					validation_DataLoader=val_loader,
					lr_scheduler=scheduler,
					epochs=params['epochs'],
					logger=run,
					tuner=tuner,
					)

	# start training
	training_losses, validation_losses, lr_rates = trainer.run_trainer()

	run['learning_rates'].log(lr_rates)
	
	if save:
		model_name = save
		torch.save(model.state_dict(), model_name)
		# run['model/weights'].upload(model_name)

	losses = test(model, dataset_path, dataset_name, test_data, run=run, criterion=criterion, device=device, num_workers=num_workers, label_size=label_size)
	run['test/df'].upload(File.as_html(losses))
	# run['test/test'].log(losses) #if dict

	run.stop()


def test(model, dataset_path, dataset_name, test_set, threshold=0.5, num_workers=4, batch_size=1, criterion=torch.nn.BCEWithLogitsLoss(), run=False, device='cpu', label_size:tuple=(64,64,64)):
	# TODO add detection r param
	
	dc = DeepColloid(dataset_path)
	print('Running test, this may take a while...')
	
	# test on real data
	real_dict = read_real_examples()
	for name, d in real_dict.items():
		pred_positions, label = dc.detect(d['array'], diameter = 7, model=model, debug=True)
		print(pred_positions)
		if len(pred_positions>0):
			sidebyside = make_proj(d['array'], label)
			run[name].upload(File.as_image(sidebyside))

			trackpy_pos = run_trackpy(d['array'], diameter = dc.round_up_to_odd(d['diameter'])-2)
			x, y = dc.get_gr(trackpy_pos, 50, 100)
			plt.plot(x, y, label=f'tp n ={len(trackpy_pos)}', color='gray')
			x, y = dc.get_gr(pred_positions, 50, 100)
			plt.plot(x, y, label=f'unet n ={len(pred_positions)}', color='red')
			plt.legend()
			fig = plt.gcf()
			run[name+'gr'].upload(fig)
			plt.clf()

	# test predict on sim
	data_dict = dc.read_hdf5('test', 1)
	test_array, true_positions, label, diameters, metadata = data_dict['image'], data_dict['positions'], data_dict['label'], data_dict['diameters'], data_dict['metadata']
	pred_positions, test_label = dc.detect(test_array, diameter = 7, model=model, debug=True)
	sidebyside = make_proj(test_array, test_label)
	run['prediction'].upload(File.as_image(sidebyside))
	trackpy_positions = dc.run_trackpy(test_array, dc.round_up_to_odd(metadata['params']['r']*2))


	if len(pred_positions) == 0:
		print('Skipping gr() as bad pred')
	else:
		x, y = dc.get_gr(true_positions, 50, 100)
		plt.plot(x, y, label=f'true n ={len(true_positions)}', color='gray')
		x, y = dc.get_gr(pred_positions, 50, 100)
		plt.plot(x, y, label=f'unet n ={len(pred_positions)}', color='red')

		x, y = dc.get_gr(trackpy_positions, 50, 100)
		plt.plot(x, y, label=f'trackpy n ={len(trackpy_positions)}', color='black')
		plt.legend()
		fig = plt.gcf()
		run['gr'].upload(fig)
		plt.clf()

	ap, precisions, recalls, thresholds = dc.average_precision(true_positions, pred_positions, diameters=diameters)
	run['AP'] = ap
	fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='Unet', tag='o-', color='red')
	ap, precisions, recalls, thresholds = dc.average_precision(true_positions, trackpy_positions, diameters=diameters)
	fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='trackpy', tag='x-', color='gray')

	run['PR_curve'].upload(fig)
	plt.clf()

	test_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, test_set, return_metadata=False, label_size=label_size) 
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

	losses = []
	model.eval()
	with torch.no_grad():
		for idx, batch in enumerate(test_loader):
			i = test_set[idx]
			metadata, true_positions, diameters = dc.read_metadata(dataset_name, i)
			true_positions, diameters = crop_positions_for_label(true_positions, label_size, diameters=diameters)

			x, y = batch
			x, y = x.to(device), y.to(device)
			# print(x.shape, x.max(), x.min())
			# print(y.shape, y.max(), y.min())

			#TODO make this dependant on criterion?

			out = model(x)  # send through model/network
			loss = criterion(out, y)
			loss = loss.cpu().numpy()
			out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits
			# post process to numpy array
			result = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
			result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

			pred_positions = find_positions(result, threshold)
			pred_positions = run_trackpy(result, diameter=dc.round_up_to_odd(metadata['params']['r']*2))
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

			losses.append(m)

	losses = pd.DataFrame(losses)

	print(losses)
	fig, axs = plt.subplots(3,2)
	sns.scatterplot(x='volfrac', y = 'precision', data=losses, ax=axs[0,0])
	sns.scatterplot(x='noise', y = 'precision', data=losses, ax=axs[0,1])
	sns.scatterplot(x='psf_zoom', y = 'precision', data=losses, ax=axs[1,0])
	sns.scatterplot(x='brightness', y = 'precision', data=losses, ax=axs[1,1])
	sns.scatterplot(x='r', y = 'precision', data=losses, ax=axs[2,1])
	fig.tight_layout()
	run['test/params_vs_prec'].upload(fig)

	plt.clf()
	fig, axs = plt.subplots(3,2)
	sns.scatterplot(x='volfrac', y = 'recall', data=losses, ax=axs[0,0])
	sns.scatterplot(x='noise', y = 'recall', data=losses, ax=axs[0,1])
	sns.scatterplot(x='psf_zoom', y = 'recall', data=losses, ax=axs[1,0])
	sns.scatterplot(x='brightness', y = 'recall', data=losses, ax=axs[1,1])
	sns.scatterplot(x='r', y = 'recall', data=losses, ax=axs[2,1])
	fig.tight_layout()
	run['test/params_vs_rec'].upload(fig)

	plt.clf()
	fig, axs = plt.subplots(3,2)
	sns.scatterplot(x='volfrac', y = 'loss', data=losses, ax=axs[0,0])
	sns.scatterplot(x='noise', y = 'loss', data=losses, ax=axs[0,1])
	sns.scatterplot(x='psf_zoom', y = 'loss', data=losses, ax=axs[1,0])
	sns.scatterplot(x='brightness', y = 'loss', data=losses, ax=axs[1,1])
	sns.scatterplot(x='r', y = 'loss', data=losses, ax=axs[2,1])
	fig.tight_layout()
	run['test/params_vs_loss'].upload(fig)

	return losses



"""
Trainer
"""

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
					self.lr_scheduler.step(self.validation_loss[i])  # learning rate scheduler step with validation loss
				else:
					self.lr_scheduler.step()  # learning rate scheduler step
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

			# print(input_.shape, out.shape, target.shape)
			# print(out.max(), target.max())
			# out_sigmoid = sig(out)

			loss = self.criterion(out, target)  # calculate loss
			loss_value = loss.item() # .item? for other losses
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



"""
LR finder
"""


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

"""
Utils
"""

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

def read_real_examples():

	d = {}

	d['abraham'] = {}
	d['abraham']['diameter'] = 15
	d['abraham']['array'] = io.imread('examples/Data/abraham.tiff')
	d['emily'] = {}
	d['emily']['diameter'] = 9
	d['emily']['array'] = io.imread('examples/Data/emily.tiff')
	d['katherine'] = {}
	d['katherine']['diameter'] = 9
	d['katherine']['array'] = io.imread('examples/Data/katherine.tiff')
	d['levke'] = {}
	d['levke']['diameter'] = 9
	d['levke']['array'] = io.imread('examples/Data/levke.tiff')

	return d

def run_trackpy(array, diameter=5, *args, **kwargs):
	df = None
	df = tp.locate(array, diameter=5, *args, **kwargs)
	f = list(zip(df['z'], df['y'], df['x']))
	tp_predictions = np.array(f)

	return tp_predictions

def renormalise(tensor: torch.Tensor):
	array = tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	array = np.squeeze(array)  # remove batch dim and channel dim -> [H, W]
	array = array * 255
	return array

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def make_proj(test_array, test_label):

	array_projection = np.max(test_array, axis=0)
	label_projection = np.max(test_label, axis=0)*255
	# new_label = np.zeros_like(array_projection)

	if label_projection.shape != array_projection.shape:
		label_projection.resize(array_projection.shape)

	sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	sidebyside /= sidebyside.max()

	return sidebyside

