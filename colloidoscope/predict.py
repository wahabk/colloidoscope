import scipy
import torch
import numpy as np

import trackpy as tp
from .models.unet import UNet
import pandas as pd
import torchio as tio
import monai
from tqdm import tqdm
from pathlib2 import Path

from typing import Union

import copy

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
	label[label > threshold] = 255
	label[label < threshold] = 0
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

def detect(array:np.ndarray, diameter:Union[int, list]=5, model:torch.nn.Module=None, weights_path:Union[str, Path] = None, 
			patch_overlap:tuple=(16, 16, 16), roiSize:tuple=(64,64,64), debug:bool=False) -> pd.DataFrame:
	"""Detect 3d spheres from confocal microscopy

	Args:
		array (np.ndarray): Image for particles to be detected from.
		diameter (Union[int, list], optional): Diameter of particles to feed to TrackPy, can be int or list the same length as image dimensions. Defaults to 5.
		model (torch.nn.Module, optional): Pytorch model. Defaults to None.
		weights_path (Union[str, Path], optional): Path to model weights file. Defaults to None.
		patch_overlap (tuple, optional): Overlap for patch based inference, overlap must be diff between input and output shape (if they are not the same). Defaults to (16, 16, 16).
		roiSize (tuple, optional): Size of ROI for model. Defaults to (64,64,64).
		debug (bool, optional): Option to return model output and positions in format for testing. Defaults to False.

	Returns:
		pd.DataFrame: TrackPy positions dataframe
	"""	

	# TODO write asserts
	
	# initialise torch device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

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

	model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model

	model = model.to(device)
	array = np.array(array/array.max(), dtype=np.float32) # normalise input
	array = np.expand_dims(array, 0) # add batch axis
	tensor = torch.from_numpy(array)
	# tensor = tensor.unsqueeze(1)

	# print(tensor.shape, tensor.max(), tensor.min())
	# print(path)

	# TODO NORMALISE BRIGHTNESS HISTOGRAM BEFORE PREDICITON
	subject_dict = {'scan' : tio.ScalarImage(tensor=array, type=tio.INTENSITY, path=None),}
	subject = tio.Subject(subject_dict) # use torchio subject to enable using grid sampling
	grid_sampler = tio.inference.GridSampler(subject, patch_size=roiSize, patch_overlap=patch_overlap, padding_mode='mean')
	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
	aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop') # average for bc, crop for normal
	# TODO make put in center like for torch
	
	model.eval()
	with torch.no_grad():
		for i, patch_batch in tqdm(enumerate(patch_loader)):
			input_tensor = patch_batch['scan'][tio.DATA]
			locations = patch_batch[tio.LOCATION]
			input_tensor.to(device)
			out = model(input_tensor)  # send through model/network
			out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits
			# print(out_sigmoid.shape, input_tensor.shape)
			
			# blank = torch.zeros_like(input_tensor) # because tio doesnt accept outputs of different sizes
			# out_sigmoid = insert_in_center(blank, out_sigmoid)

			aggregator.add_batch(out_sigmoid, locations)

	output_tensor = aggregator.get_output_tensor()
	# post process to numpy array
	result = output_tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

	# find positions from label
	# TODO change to trackpy or watershed?

	positions = run_trackpy(result*255, diameter=diameter)

	# positions = find_positions(result, threshold)

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

def run_trackpy(array, diameter=5, *args, **kwargs):
	df = None
	df = tp.locate(array, diameter=diameter, *args, **kwargs)
	f = list(zip(df['z'], df['y'], df['x']))
	tp_predictions = np.array(f, dtype='float32')

	return tp_predictions