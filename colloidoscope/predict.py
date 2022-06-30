import scipy
import torch
import numpy as np

import trackpy as tp
from .models.unet import UNet
import pandas as pd
import torchio as tio
import monai

def predict(scan, model, device='cpu', weights_path=None, threshold=0.5, return_positions=False):
	
	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location='cpu') # read trained weights
		# print(model_weights.keys())
		model.load_state_dict(model_weights) # add weights to model

	array = scan.copy()
	array = np.array(array/255, dtype=np.float32)
	array = np.expand_dims(array, 0)      # add channel axis
	array = np.expand_dims(array, 0)      # add batch axis
	array = torch.from_numpy(array)
	array = array.to(device)  # to torch, send to device
	
	model.eval()
	with torch.no_grad():
		out = model(array)  # send through model/network

	out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits
	out_relu = torch.relu(out)
	
	# post process to numpy array
	result = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(result)  # remove batch dim and channel dim -> [H, W]

	if return_positions:
		positions = find_positions(result, threshold)
		return result, positions
	else:
		return result

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

def put_in_center_like(test_array, test_label):
	new_label = np.zeros_like(test_array)
	a = test_array.shape[0]
	l = test_label.shape[0]
	diff = int((a-l)/2)
	print(a, l, diff)
	new_label[diff:a-diff, diff:a-diff, diff:a-diff] = test_label
	return new_label

def detect(array, diameter=5, model=None, patch_overlap=(16, 16, 16), roiSize=(64,64,64), threshold = 0.5, weights_path = None, debug=False):
	"""
	overlap must be diff between input and output shape (if they are not the same)
	"""
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# weights_path = 'output/weights/unet.pt'
	# device = torch.device("cpu")	
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

	model = torch.nn.DataParallel(model, device_ids=None)

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		# print(model_weights.keys())
		model.load_state_dict(model_weights) # add weights to model

	model = model.to(device)
	
	array = array.copy()
	array = np.array(array/array.max(), dtype=np.float32)
	array = np.expand_dims(array, 0)      # add batch axis
	# array = np.expand_dims(array, 0)      # add channel axis ?
	# array = torch.from_numpy(array)

	# TODO NORMALISE BRIGHTNESS HISTOGRAM BEFORE PREDICITON
	subject = tio.Subject(scan = tio.ScalarImage(tensor=array)) # use torchio subject to enable using grid sampling
	grid_sampler = tio.inference.GridSampler(subject, patch_size=roiSize, patch_overlap=patch_overlap, padding_mode='mean')
	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
	aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop') # average for bc
	
	model.eval()
	with torch.no_grad():
		for patch_batch in patch_loader:
			input_tensor = patch_batch['scan'][tio.DATA]
			locations = patch_batch[tio.LOCATION]
			input_tensor.to(device)
			out = model(input_tensor)  # send through model/network
			out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits					
			# print(out_sigmoid.shape, input_tensor.shape)
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