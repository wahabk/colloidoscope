import torch
import numpy as np
from src.trainer import Trainer, plot_training, LearningRateFinder
from src.unet import UNet
import torchio as tio
from src.dataset import ColloidsDatasetSimulated
from src.deepcolloid import DeepColloid
import matplotlib.pyplot as plt
import neptune.new as neptune
import scipy

def predict(array, threshold, model, weights_path, device, return_positions=False):
	
	
	model_weights = torch.load(weights_path) # read trained weights
	model.load_state_dict(model_weights) # add weights to model

	print(array.shape)
	array = np.array(array/255, dtype=np.float32)
	array = np.expand_dims(array, 0)      # add channel axis
	array = np.expand_dims(array, 0)      # add batch axis
	array = torch.from_numpy(array).to(device)  # to torch, send to device
	print(array.shape)
	
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
		return label, positions
	else:
		return label


dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/home/wahab/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)
# dc = DeepColloid(dataset_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'predict on {device}')

# model
model = UNet(in_channels=1,
			 out_channels=1,
			 n_blocks=4,
			 start_filters=32,
			 activation='relu',
			 normalization='batch',
			 conv_mode='same',
			 dim=3).to(device)

roiSize = (32,128,128)
batch_size = 2
num_workers = 2
threshold = 0.3
weights_path =  'output/weights/unet.pt'
dataset_name = 'replicate'

array = dc.read_hdf5(dataset_name, 45)

label, positions = predict(array, threshold, model, weights_path, device, return_positions=True)

dc.make_gif(label, 'output/pytorch_predict.gif')