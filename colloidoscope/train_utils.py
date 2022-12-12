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
# from .models.unet import UNet
# from .simulator import crop_pos_for_test
from .predict import *
from torch.nn import BCELoss
import torch.nn.functional as F

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


from functools import reduce

from skimage.feature import peak_local_max, blob_log

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

		# TODO use tio subject to avoid this

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
				 transformer=False,
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
		self.transformer = transformer

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
			# print(out.shape)
			if self.transformer: 
				try:
					out = out[0]
				except:
					print(out.shape)
					
			if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) == False:
				out = torch.sigmoid(out)
				loss = self.criterion(out, target)
			else:
				loss = self.criterion(out, target)
				out = torch.sigmoid(out)
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
				if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) == False:
					out = torch.sigmoid(out)
					loss = self.criterion(out, target)
				else:
					loss = self.criterion(out, target)
					out = torch.sigmoid(out)
				loss_value = loss.item()
				valid_losses.append(loss_value)
				if self.logger: self.logger['val/loss'].log(loss_value)

				batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

		self.validation_loss.append(np.mean(valid_losses))
		if self.tuner: tune.report(val_loss=(np.mean(valid_losses)))


		batch_iter.close()


"""
Test
"""

def exclude_borders(centers, canvas_size, pad, diameters=None, label_size=None):
	if label_size is not None:
		zdiff = (canvas_size[0] - label_size[0])/2
		xdiff = (canvas_size[1] - label_size[1])/2
		ydiff = (canvas_size[2] - label_size[2])/2
		centers = centers - [zdiff, xdiff, ydiff]

	indices = []
	for idx, c in enumerate(centers):
		if pad<=c[0]<=(canvas_size[0]-pad) and pad<=c[1]<=(canvas_size[1]-pad) and pad<=c[2]<=(canvas_size[2]-pad):
			indices.append(idx)

	final_centers = centers[indices]

	if diameters is not None:
		final_diameters = diameters[indices]
		return final_centers, final_diameters
	else:
		return final_centers

def plot_gr(x, y, diameter, label=f'prediction', color='gray', axs=None, fontsize='medium'):
	if isinstance(diameter, list): diameter = min(diameter)
	x = x / diameter
	if axs is None:
		plt.plot(x, y, label=label, color=color)
		plt.xlabel("$r / \sigma$", fontsize=fontsize)
		plt.ylabel("$g(r)$", fontsize=fontsize)
		plt.xticks(list(range(0,6)))
		plt.xlim((0,5))
		plt.legend()
	else:
		axs.plot(x, y, label=label, color=color)
		axs.set_xlabel("$r / \sigma$", fontsize=fontsize)
		axs.set_ylabel("$g(r)$", fontsize=fontsize)
		axs.set_xticks(list(range(0,6)))
		axs.set_xlim((0,5))
	return

def frac_detected(target_volfrac, radius, n_particles, image_size):
	single_vol = (4/3) * np.pi * radius**3
	measured_volume = n_particles * single_vol
	measured_volfrac = measured_volume / image_size
	fraction_detected = measured_volfrac / target_volfrac
	return fraction_detected

def test(model, dataset_path, dataset_name, test_set, threshold=0.5, 
		num_workers=4, batch_size=4, criterion=torch.nn.BCEWithLogitsLoss(), 
		run=False, device='cpu', canvas_size:tuple=(64,64,64), label_size:tuple=(64,64,64), 
		heatmap_r="radius", post_processing="tp", work_dir=None):
	
	dc = DeepColloid(dataset_path)
	print('Running test, this may take a while...')

	os.chdir(work_dir)
	dataset_name = dataset_name+"_test"
	
	# test on real data
	real_dict = read_real_examples()

	real_len = len(real_dict)
	fig, axs = plt.subplots(real_len,3,)
	plt.tight_layout(pad=0.01)

	for i, (name, d) in enumerate(real_dict.items()):
		print(name, d['array'].shape, i, real_len)

		if heatmap_r == "radius":
			detection_diameter = d['diameter']
		else:
			detection_diameter = heatmap_r

		print(canvas_size,   label_size)
		df, pred_positions, label = dc.detect(d['array'], diameter = detection_diameter, model=model, 
									roiSize=canvas_size, label_size=label_size,
									debug=True, post_processing=post_processing, run_on="cuda", 
									remove_borders=True)

		trackpy_pos, df = dc.run_trackpy(d['array'], diameter = detection_diameter)
		trackpy_pos = exclude_borders(trackpy_pos, d['array'].shape, pad=d['diameter'][0]/2)
		
		tp_frac_detected = frac_detected(d['volfrac'], max(d['diameter'])/2, len(trackpy_pos), d['array'].size)
		unet_frac_detected = frac_detected(d['volfrac'], max(d['diameter'])/2, len(pred_positions), d['array'].size)

		if len(pred_positions)>0:
			array_projection = np.max(d['array'], axis=0)
			label_projection = np.max(label, axis=0)*255
			axs[i,0].imshow(array_projection, cmap='gray')
			axs[i,0].set_xticks([])
			axs[i,0].set_yticks([])
			axs[i,0].set_title(name, fontsize="xx-large")
			axs[i,1].imshow(label_projection, cmap='gist_heat')
			axs[i,1].set_xticks([])
			axs[i,1].set_yticks([])
			#TODO add zoom in co			

			x, y = dc.get_gr(trackpy_pos, 100, 100)
			plot_gr(x, y, d['diameter'], label=f'TP n ={len(trackpy_pos)}(~{tp_frac_detected*100:.0f}%)', color='gray', axs=axs[i,2], fontsize='x-large')
			x, y = dc.get_gr(pred_positions, 100, 100)
			plot_gr(x, y, d['diameter'], label=f'Unet n ={len(pred_positions)}(~{unet_frac_detected*100:.0f}%)', color='red', axs=axs[i,2], fontsize='x-large')
			if i != real_len-1:
				axs[i,2].set_xlabel("")
				axs[i,2].set_xticks([])
			axs[i,2].legend(loc='lower right', fontsize='large')

		else:
			print('\n\n\nNOT DETECTING PARTICLES\n\n\n')

	fig.set_figwidth(3*4)
	fig.set_figheight(real_len*4)
	if run: run['grs'].upload(fig)
	plt.clf()

	
	# test predict on sim
	data_dict = dc.read_hdf5(dataset_name, 10000)
	test_array, true_positions, label, diameters, metadata = data_dict['image'], data_dict['positions'], data_dict['label'], data_dict['diameters'], data_dict['metadata']
	# TODO exclude borders in ap?
	print(test_array.shape, test_array.dtype, test_array.min(), test_array.max(), test_array.mean())

	if heatmap_r == "radius":
		detection_diameter = dc.round_up_to_odd(metadata['params']['r']*2)
	else:
		detection_diameter = heatmap_r
	if post_processing == "log":
		#TODO add diameters as array
		r = metadata['params']['r']
		detection_diameter = r*2

	df, pred_positions, test_label = dc.detect(test_array, diameter = detection_diameter, model=model, 
												roiSize=canvas_size, label_size=label_size,
												debug=True, post_processing=post_processing,  run_on="cuda",
												remove_borders=True)
	test_label = np.array(test_label*255, dtype="uint8")
	print(test_array.shape, test_array.dtype, test_array.min(), test_array.max(), test_array.mean())
	print(test_label.shape, test_label.dtype, test_label.min(), test_label.max(), test_label.mean())
	# test_array = (test_array/test_array.max())*255
	# test_label = (test_label/test_label.max())*255

	trackpy_pos, df = dc.run_trackpy(test_array, dc.round_up_to_odd(metadata['params']['r']*2))

	true_positions, diameters = exclude_borders(true_positions, canvas_size=test_array.shape, pad=metadata['params']['r']*1, diameters=diameters)
	trackpy_pos = exclude_borders(trackpy_pos, test_array.shape, pad=metadata['params']['r']*1)

	# biggr fig
	fig, axs = plt.subplots(1,4)
	plt.tight_layout(pad=0.1)

	array_projection = np.max(test_array, axis=0)
	label_projection = np.max(test_label, axis=0)
	array_projection = ndimage.zoom(array_projection, 2)
	label_projection = ndimage.zoom(label_projection, 2)
	axs[0].imshow(array_projection, cmap='gray')
	axs[0].set_xticks([])
	axs[0].set_yticks([])
	axs[0].set_title("Image", fontsize="large")
	axs[1].imshow(label_projection, cmap='gist_heat')
	axs[1].set_xticks([])
	axs[1].set_yticks([])
	axs[1].set_title("Prediction", fontsize="large")

	try:
		x, y = dc.get_gr(true_positions, 100, 100)
		plot_gr(x, y, diameter=(metadata['params']['r']*2), label=f'True n ={len(true_positions)}', axs=axs[2], color='gray')
		x, y = dc.get_gr(pred_positions, 100, 100)
		plot_gr(x, y, diameter=(metadata['params']['r']*2), label=f'Unet n ={len(pred_positions)}', axs=axs[2], color='red')
		x, y = dc.get_gr(trackpy_pos, 100, 100)
		plot_gr(x, y, diameter=(metadata['params']['r']*2), label=f'TP n ={len(trackpy_pos)}', axs=axs[2], color='black')
		axs[2].legend()
	except:
		print('Skipping gr() as bad pred')
		if run: run['gr'] = 'failed'
 
	ap, 	precisions, recalls, thresholds = dc.average_precision(true_positions, pred_positions, diameters=diameters)
	dc.plot_pr(ap, precisions, recalls, thresholds, name='Unet', tag='o-', color='red', axs=axs[3])
	tp_ap, 	precisions, recalls, thresholds = dc.average_precision(true_positions, trackpy_pos, diameters=diameters)
	dc.plot_pr(ap, precisions, recalls, thresholds, name='TP', tag='x-', color='gray', axs=axs[3])

	fig.set_figwidth(12)
	fig.set_figheight(2)
	if run: run['AP'] = ap
	if run: run['bigGR'].upload(fig)

	test_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, test_set, label_size=label_size, transform=None) 
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

	losses = []
	model.eval()
	with torch.no_grad():
		for batch_idx, batch in enumerate(test_loader):
			print(batch_idx, "/", int(len(test_set)/batch_size))
			
			x, y = batch
			inputs = x.clone()
			x, y = x.to(device), y.to(device)

			out = model(x)  # send through model/network
			if isinstance(criterion, torch.nn.BCEWithLogitsLoss) == False:
				out = torch.sigmoid(out)
				loss = criterion(out, y)
			else:
				# TODO raise Exception("Figure out how to run sigmoid for other LFs than BCEWithLogits?")
				loss = criterion(out, y)
				out = torch.sigmoid(out)
			loss = loss.cpu().numpy()
			# post process to numpy array
			results = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray
			inputs = inputs.cpu().numpy()

			for test_idx, [result, for_tp] in tqdm(enumerate(zip(results, inputs))):
				i = test_idx + ((batch_idx)*(batch_size))
				index = test_set[i]
				metadata, true_positions, diameters = dc.read_metadata(dataset_name, index)
				result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

				if heatmap_r == "radius":
					detection_diameter = dc.round_up_to_odd(metadata['params']['r']*2)
				else:
					detection_diameter = heatmap_r
				if post_processing == "tp":
					pred_positions, _ = dc.run_trackpy(result, diameter=detection_diameter)
				elif post_processing == "max":
					max_diameter = int((metadata['params']['r']*2))
					pred_positions = peak_local_max(result*255, min_distance=max_diameter)
				if post_processing == "log":
					# result[result<threshold] = 0
					diameter = metadata['params']['r']*2
					sigma = (diameter/2)/math.sqrt(3)
					max_sigma = (diameter*2)/math.sqrt(3)
					pred_positions = blob_log(result*255, min_sigma=sigma, max_sigma=max_sigma, overlap=0)[:,:-1]
				
				array = dc.crop3d(np.squeeze(for_tp), roiSize=label_size)
				detection_diameter  =  dc.round_up_to_odd(metadata['params']['r']*2)
				tp_positions, _ = dc.run_trackpy(array, diameter=detection_diameter)

				true_positions, diameters = exclude_borders(true_positions, canvas_size=canvas_size, label_size=label_size,
															pad=metadata['params']['r']*1, diameters=diameters)
				pred_positions = exclude_borders(pred_positions, label_size, pad=metadata['params']['r']*1)
				tp_positions = exclude_borders(tp_positions, label_size, pad=metadata['params']['r']*1)
				

				prec, rec = dc.get_precision_recall(true_positions, pred_positions, diameters, 0.5,)
				tp_prec, tp_rec = dc.get_precision_recall(true_positions, tp_positions, diameters, 0.5,)


				m = {
					'dataset'		: metadata['dataset'],
					'n' 	 		: metadata['n'],
					'idx'	 		: index,
					'volfrac'		: metadata['volfrac'],
					'n_particles'	: metadata['n_particles'],
					'type'			: metadata['type'],
					**metadata['params'],
					'loss'	 		: float(loss),
					'precision'	 	: float(prec),
					'recall'	 	: float(rec),
					'tp_precision'	: float(tp_prec),
					'tp_recall'	 	: float(tp_rec),
				}

				losses.append(m)

	losses = pd.DataFrame(losses)
	print(losses)

	plot_params = ['volfrac', 'snr', 'cnr', 'particle_size', 'brightness', 'r']
	titles		= ['$\phi$', 'SNR', 'CNR', 'Size ($\mu m$)', '$f_\mu$ (0-255)', 'Radius (pxls)']

	fig,axs = plt.subplots(2,len(plot_params),  sharey=True)
	plt.tight_layout(pad=0)

	this_axs = axs[0,:].flatten()
	for i, p in enumerate(plot_params):
		this_df = losses[losses['type'].isin([p])]
		this_axs[i].scatter(x=p, 		y = 'tp_precision', data=this_df, color='black', marker='<')
		this_axs[i].scatter(x=p, 		y = 'precision', 	data=this_df, color='red', marker='>')
		this_axs[i].set_xticks([])
		if i == 0: 
			this_axs[i].set_ylabel("Precision", fontsize='large')
			this_axs[i].set_yticks([0,0.25,0.5,0.75,1])
			this_axs[i].set_ylim(-0.1,1.1)

	this_axs = axs[1,:].flatten()
	for i, p in enumerate(plot_params):
		this_df = losses[losses['type'].isin([p])]
		this_axs[i].scatter(x=p, 		y = 'tp_recall', data=this_df, color='black', marker='<')
		this_axs[i].scatter(x=p, 		y = 'recall', 	data=this_df, color='red', marker='>')
		this_axs[i].set_xlabel(titles[i], fontsize='large')
		if i == 0: 
			this_axs[i].set_ylabel("Recall", fontsize='large')

	fig.set_figwidth(13)
	fig.suptitle("Precisions and Recalls", fontsize='xx-large', y=1.05)
	if run: run['test/params_vs_PR'].upload(fig)

	fig, axs = plt.subplots(1,len(plot_params), sharey="row")
	fig.tight_layout(pad=0)
	axs = axs.flatten()
	for i, p in enumerate(plot_params):
		this_df = losses[losses['type'].isin([p])]
		sns.scatterplot(x=p, 		y = 'loss', data=this_df, ax=axs[i])
		axs[i].scatter(x=p, 		y = 'loss', data=this_df, color='blue')
		axs[i].set_xlabel(titles[i], fontsize='large')
		if i == 0: 
			this_axs[i].set_ylabel("Loss", fontsize='large')
			this_axs[i].set_yticks([0,0.25,0.5,0.75,1])
		else:
			this_axs[i].set_yticks([])
	plt.yscale('log')
	plt.ylim(pow(10,-2),pow(10,0))
	fig.suptitle("Loss", y=1.1, fontsize='xx-large')
	fig.set_figheight(2)
	fig.set_figwidth(13)
	if run: run['test/params_vs_loss'].upload(fig)

	return losses

def myRound(n):
	answer = round(n)
	if not answer%2:
		return answer
	return answer + 1

def get_subplot_indices(n_figs:int):
	n_figs = myRound(n_figs)
	# get factors
	factors = list(reduce(list.__add__, 
					([i, n_figs//i] for i in range(1, int(n_figs**0.5) + 1) if n_figs % i == 0)))
	factors.sort()
	# print(factors)

	if (len(factors) % 2) == 0: # if even
		indx = int(len(factors)/2)
		first_middle_factor = factors[indx-1]
		second_middle_factor = factors[indx]

		fig, axs = plt.subplots(first_middle_factor, second_middle_factor)

		plt_indices = [i for i in np.ndindex((first_middle_factor,second_middle_factor))]
		# print(plt_indices)
		
	else:
		indx = math.floor(len(factors)/2)
		factor = factors[indx]

		fig, axs = plt.subplots(factor, factor)

		plt_indices =  list(np.ndindex((factor,factor)))

	return fig, axs, plt_indices

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

	# d['abraham'] = {}
	# d['abraham']['diameter'] = [13,11,11]
	# d['abraham']['array'] = io.imread('examples/Data/abraham.tiff')
	d['emily'] = {}
	d['emily']['diameter'] = [15,9,9]
	d['emily']['volfrac'] = 0.5
	d['emily']['array'] = io.imread('examples/Data/emily.tiff')
	# d['katherine'] = {}
	# d['katherine']['diameter'] = [7,7,7]
	# d['katherine']['array'] = io.imread('examples/Data/katherine.tiff')
	d['levke'] = {}
	d['levke']['diameter'] = [15,11,11]
	d['levke']['volfrac'] = 0.58
	d['levke']['array'] = io.imread('examples/Data/levke.tiff')
	d['james'] = {}
	d['james']['diameter'] = [17,15,15]
	d['james']['volfrac'] = 0.58
	d['james']['array'] = io.imread('examples/Data/james.tiff')
	d['jamesdecon'] = {}
	d['jamesdecon']['diameter'] = [17,15,15]
	d['jamesdecon']['volfrac'] = 0.58
	d['jamesdecon']['array'] = io.imread('examples/Data/jamesdecon.tiff')

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

	# TODO add scale bars

	array_projection = np.max(test_array, axis=0)
	label_projection = np.max(test_label, axis=0)*255
	# new_label = np.zeros_like(array_projection)

	if label_projection.shape != array_projection.shape:
		label_projection.resize(array_projection.shape)

	sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	sidebyside /= sidebyside.max()

	return sidebyside
