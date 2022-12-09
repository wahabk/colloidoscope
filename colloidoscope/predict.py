import scipy
import torch
import numpy as np

import trackpy as tp
import pandas as pd
import torchio as tio
import monai
from tqdm import tqdm
from pathlib2 import Path

from monai.data import GridPatchDataset, DataLoader, PatchIter

from typing import Union

import math

import copy

import torch.nn.functional as  F
from scipy import ndimage


from skimage.feature import peak_local_max, blob_dog, blob_log

def find_positions(result, threshold) -> np.ndarray:
	label = result.copy()
	# print(label.shape, label.max(), label.min())
	label = np.array(label, dtype='float32')
	# label = scipy.ndimage.zoom(label, 2, mode='nearest')
	# print(label.shape, label.max(), label.min())

	label[label > threshold] = 255
	label[label < threshold] = 0
	label = np.array(label, dtype='uint8')
	# label = scipy.ndimage.gaussian_filter(label, (2,2,2))
	print(label.shape, label.max(), label.min())
	label = np.array(label, dtype='float32')
	label = label/label.max()

	print(label.shape, label.max(), label.min())
	


	# label = scipy.ndimage.zoom(label, 0.5, mode='nearest')
	# label[label > threshold] = 255
	# label[label < threshold] = 0
	print(label.shape, label.max(), label.min())

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

def insert_in_center(a:np.ndarray, b:np.ndarray):

	_batch, _channels, roiZ, roiY, roiX = b.shape
	zl = int(roiZ / 2) # compute lens
	yl = int(roiY / 2)
	xl = int(roiX / 2)

	a_center = [int(a.shape[2] / 2), int(a.shape[3] / 2), int(a.shape[4] / 2)]
	z, y, x = a_center
	z, y, x = int(z), int(y), int(x)

	a[0, 0, z - zl : z + zl, y - yl : y + yl, x - xl : x + xl] = b

	return a


def run_trackpy(array, diameter=5, *args, **kwargs):
	df = None
	df = tp.locate(array, diameter=diameter, *args, **kwargs)
	f = list(zip(df['z'], df['y'], df['x']))
	tp_predictions = np.array(f, dtype='float32')

	return tp_predictions

def detect(array:np.ndarray, diameter:Union[int, list]=1, model:torch.nn.Module=None, weights_path:Union[str, Path] = None, 
			patch_overlap:tuple=(16, 16, 16), roiSize:tuple=(64,64,64), label_size:tuple=(60,60,60), post_processing:str="tp", threshold:float=0.5, 
			debug:bool=False, run_on="cpu", batch_size=4) -> pd.DataFrame:
	"""Detect 3d spheres from confocal microscopy

	Args:
		array (np.ndarray): Image for particles to be detected from.
		diameter (Union[int, list], optional): Diameter of particles to feed to TrackPy, can be int or list the same length as image dimensions. Defaults to 5. If post_processing == str(max) this has to be int it will be min_distance.
		model (torch.nn.Module, optional): Pytorch model. Defaults to None.
		weights_path (Union[str, Path], optional): Path to model weights file. Defaults to None.
		patch_overlap (tuple, optional): Overlap for patch based inference, overlap must be diff between input and output shape (if they are not the same). Defaults to (16, 16, 16).
		roiSize (tuple, optional): Size of ROI for model. Defaults to (64,64,64).
		debug (bool, optional): Option to return model output and positions in format for testing. Defaults to False.
		post_processing (str, optional): one of ["tp", "max", "log"]
		run_on (str, optional): Which device to run on, can be "cpu" or "cuda"

	Returns:
		pd.DataFrame: TrackPy positions dataframe
	"""	

	# TODO write asserts

	if post_processing not in ["tp", "max", "log"]:
		raise ValueError(f"post_processing can be str(tp), or str(max) but you provided {post_processing}")
	
	# initialise torch device
	if run_on not in ["cpu", "cuda"]:
		raise ValueError(f"You gave run_on={run_on} but it can only be cuda or cpu")
	elif run_on == "cuda" and torch.cuda.is_available() == False:
		raise ValueError("You gave run_on='cuda' but cuda isnt available, check torch installation")
	
	print(f"Requested to run on {run_on}")
	device = torch.device(run_on)
	print(f'Predicting on {device}')

	# model
	if model is None:
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			channels=[32, 64, 128],
			strides=[2,2],
			# act=params['activation'],
			# norm=params["norm"],
			padding='valid',
		)
	
	if 'attention_unet_202211' in weights_path:
		n_blocks=2
		start = int(math.sqrt(32))
		channels = [2**n for n in range(start, start + n_blocks)]
		strides = [2 for n in range(1, n_blocks)]
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			channels=channels,
			strides=strides,
			kernel_size=7,
			# up_kernel_size=3,
			padding='valid',
		)

	model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model
	
	if run_on == "cpu": model = model.module.to(device)
	
	array = np.array(array/array.max(), dtype=np.float32) # normalise input
	array = np.expand_dims(array, 0) # add batch axis
	tensor = torch.from_numpy(array)
	# tensor = tensor.unsqueeze(1)

	# print(tensor.shape, tensor.max(), tensor.min())
	# print(path)

	# TODO NORMALISE BRIGHTNESS HISTOGRAM BEFORE PREDICITON
	# subject_dict = {'scan' : tio.ScalarImage(tensor=tensor, type=tio.INTENSITY, path=None),}
	# subject = tio.Subject(subject_dict) # use torchio subject to enable using grid sampling
	# grid_sampler = tio.inference.GridSampler(subject, patch_size=roiSize, patch_overlap=(16,16,16), padding_mode='mean')
	# patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
	# aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop') # average for bc, crop for normal
	
	grid_sampler = MyGridSampler(tensor, patch_size=roiSize, overlap=patch_overlap, label_size=label_size)
	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
	aggregator = MyAggregator(grid_sampler, overlap_mode='avg') # average for bc, crop for normal

	# patch_iter = PatchIter(patch_size=roiSize, start_pos=(0, 0, 0))
	# ds = GridPatchDataset(data=[array],
    #                           patch_iter=patch_iter,
    #                           transform=None)
	# patch_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
	# output_tensor = np.zeros_like(array)

	model.eval()
	with torch.no_grad():
		for i, patch_batch in tqdm(enumerate(patch_loader)):
			# input_tensor = patch_batch['scan'][tio.DATA]
			# locations = patch_batch[tio.LOCATION]
			input_tensor = patch_batch['image']
			locations = patch_batch['location']
			# input_tensor, locations = patch_batch[0], patch_batch[1]
			input_tensor.to(device)
			out = model(input_tensor)  # send through model/network
			out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits
			# print(input_tensor.shape, out_sigmoid.shape, locations.shape, locations)
			# print(out_sigmoid.shape, input_tensor.shape)
			
			# blank = torch.zeros_like(input_tensor) # because tio doesnt accept outputs of different sizes
			# out_sigmoid = insert_in_center(blank, out_sigmoid)
			# out_sigmoid = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
			# for pred, loc in zip(out_sigmoid, locations):
			# 	output_tensor[	loc[0,0]:loc[0,1], 
			# 					loc[1,0]:loc[1,1], 
			# 					loc[2,0]:loc[2,1], 
			# 					loc[3,0]:loc[3,1]] = pred

			# aggregator.add_batch(out_sigmoid, locations)
			aggregator.append_batch(out_sigmoid, locations)

	# post process to numpy array
	output_tensor = aggregator.get_output_tensor()
	output_tensor = output_tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(output_tensor)  # remove batch dim and channel dim -> [H, W]
	result = (result-result.min())/result.max()

	# result = ndimage.gaussian_filter(result, (1,1,1), mode="reflect")
	result = ndimage.median_filter(result, size=2)
	# find positions from label

	if post_processing == "tp":
		positions = run_trackpy(result*255, diameter=diameter)
	elif post_processing == "max":
		if isinstance(diameter, list): diameter = diameter[0]
		positions = peak_local_max(result*255, min_distance=int((diameter/2)))
	elif post_processing == "log":
		if isinstance(diameter, list): diameter = np.array(diameter)
		# result[result<threshold] = 0
		sigma = (diameter/2)/math.sqrt(3)
		max_sigma = (diameter)/math.sqrt(3)
		positions = blob_log(result*255, min_sigma=sigma,  max_sigma=max_sigma, overlap=0)[:,:-1]
		# TODO exclude borders

	if len(positions)==0: positions = [[0,0,0]];  print("\n\n\nNOT DETECTING PARTICLES\n\n\n")



	d = {
		'x' : positions[:,1],
		'y' : positions[:,2],
		'z' : positions[:,0],
		}
	df = pd.DataFrame().from_dict(d) #, orient='index')

	if debug:
		return df, positions, result
	else:
		return df


class MyGridSampler(torch.utils.data.Dataset):
	def __init__(self, image:torch.tensor, patch_size:list, overlap:list=[0,0,0], label_size:Union[list, None]=None,) -> None:
		"""
		based on tio
		"""
		self.image = image
		self.patch_size = patch_size
		self.overlap = overlap
		self.label_size = label_size

		self.patch_tensor, self.label_smaller, self.diffs = self._make_patch_tensor(image, 
																		np.array(patch_size), 
																		np.array(overlap), 
																		np.array(label_size))
		
		self.overlap = [max(d*2, o) for d, o in zip(self.diffs, self.overlap)]

		self.locations = self._get_patches_locations(self.patch_tensor.shape[1:], self.patch_size, patch_overlap=self.overlap)


	def __len__(self) -> int:
		return len(self.locations)

	def __getitem__(self, index):
		# Assume 3D
		location = copy.deepcopy(torch.tensor(self.locations[index]))
		index_ini = location[:3]
		cropped = self.crop(self.patch_tensor, index_ini, self.patch_size)
		d = {"image":cropped,"location":location}
		return d

	def _make_patch_tensor(self, image:torch.tensor, patch_size:np.ndarray, overlap:np.ndarray, label_size:np.ndarray,):

		if label_size is None:
			patch_tensor = image
			label_smaller = False
			diffs = [0,0,0]
			raise NotImplementedError("")
			pass
		else: 
			diffs = patch_size - label_size
			diffs=diffs//2
			# tensor_size = np.array(image.shape) + ([0]+list(diffs*2))
			patch_tensor = F.pad(self.image, (diffs[0],diffs[0],diffs[1],diffs[1],diffs[2],diffs[2]), mode='reflect')
			label_smaller = True

		return patch_tensor, label_smaller, diffs

	def crop(self, image, index_ini, patch_size):
			return image[	:,
				index_ini[0]:index_ini[0]+patch_size[0], 
				index_ini[1]:index_ini[1]+patch_size[1], 
				index_ini[2]:index_ini[2]+patch_size[2]]


	def _get_patches_locations(self, image_size:list, patch_size:list, patch_overlap:list,) -> np.ndarray:
		# Example with image_size 10, patch_size 5, overlap 2:
		# [0 1 2 3 4 5 6 7 8 9]
		# [0 0 0 0 0]
		#       [1 1 1 1 1]
		#           [2 2 2 2 2]
		# Locations:
		# [[0, 5],
		#  [3, 8],
		#  [5, 10]]
		indices = []
		zipped = zip(image_size, patch_size, patch_overlap)
		for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
			end = im_size_dim + 1 - patch_size_dim
			step = patch_size_dim - patch_overlap_dim
			indices_dim = list(range(0, end, step))
			if indices_dim[-1] != im_size_dim - patch_size_dim:
				indices_dim.append(im_size_dim - patch_size_dim)
			indices.append(indices_dim)
		indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
		indices_ini = np.unique(indices_ini, axis=0)
		indices_fin = indices_ini + np.array(patch_size)
		locations = np.hstack((indices_ini, indices_fin))
		return np.array(sorted(locations.tolist()))








class MyAggregator():
	def __init__(self, sampler:MyGridSampler, overlap_mode="avg") -> None:
		self.sampler = sampler
		self.overlap_mode = overlap_mode

		self._out_tensor = torch.zeros(self.sampler.image.shape) # remember to add batch dim
		if overlap_mode=='avg': 
			self.avgmask_tensor = torch.zeros_like(self._out_tensor)


	def append_batch(self, batch_tensor, locations) -> None:

		locations[:,3:] = locations[:,3:]-(torch.tensor(self.sampler.diffs)*2)
		batch = batch_tensor.cpu()
		locations = locations.cpu().numpy()

		if self.overlap_mode == "avg":
			for patch, location in zip(batch, locations):
				i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
				self._out_tensor[
					:,
					i_ini:i_fin,
					j_ini:j_fin,
					k_ini:k_fin,
				] += patch
				self.avgmask_tensor[
					:,
					i_ini:i_fin,
					j_ini:j_fin,
					k_ini:k_fin,
				] += 1
		else: raise NotImplementedError("")


	def get_output_tensor(self,) -> torch.tensor:
		if self.sampler.label_smaller: return self._out_tensor / self.avgmask_tensor
		else: raise NotImplementedError("")