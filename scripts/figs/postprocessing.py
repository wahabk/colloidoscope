from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.predict import find_positions
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
# from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import numpy as np
import random
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path
from numba import njit
import torch

from scipy.spatial.distance import pdist, cdist

from tqdm import tqdm


from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max, blob_dog, blob_log

from math import sqrt






if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(60,60,60)


	r = 5
	d = r*2
	index = 217
	volfrac = 0.1
	threshold = 0.5


	dataset_name = 'please_1400_test'

	d = dc.read_hdf5(dataset_name, index)
	image = d['image']
	label = d["label"]
	# positions = d["positions"]
	# metadata = d["metadata"]
	# diameters = d["diameters"]
	# label[label<threshold] = 0

	image = torch.tensor(image/image.max())
	label = torch.tensor(label/label.max())

	image = image.cpu().numpy()
	label = label.cpu().numpy()

	metadata, true_positions, diameters = dc.read_metadata(dataset_name, index)
	print(metadata)

	# label = dc.crop3d(label, label_size)

	print(image.max(), image.min(), image.dtype, dc.round_up_to_odd(metadata['params']['r']*2))
	tp_pred, df = dc.run_trackpy(image, diameter = dc.round_up_to_odd(metadata['params']['r']*2), )

	# positions, diameters = crop_positions_for_label(positions, canvas_size, label_size, diameters)

	# print(label.dtype)
	# print(label.shape, label.max(), label.min())

	# distance = ndi.distance_transform_edt(np.array(label*255, dtype="uint16"))
	# coords = peak_local_max(label)
	# mask = np.zeros(distance.shape, dtype=bool)
	# mask[tuple(coords.T)] = True
	# markers, _ = ndi.label(mask)
	# new_label = watershed(-distance, markers, mask=label)

	# print(distance.shape, distance.max(), distance.min())
	# print(coords.shape, coords.max(), coords.min())
	# print(np.shape(coords))
	# print(mask.shape, mask.max(), mask.min())
	# print(new_label.shape, new_label.max(), new_label.min())

	# import pdb; pdb.set_trace()

	# coords = peak_local_max(label, min_distance=diameter/2)
	# coords = np.array(coords)
	# coords = blob_log(label, min_sigma=diameter, max_sigma=diameter, overlap=0)
	label[label<0.5] = 0
	sigma = int((metadata['params']['r'])/sqrt(3))
	# coords = blob_log(label, min_sigma=sigma, max_sigma=sigma, overlap=0)[:,:-1] # get rid of sigmas
	coords = find_positions(label, threshold=0)
	# print(len(coords))
	# print(coords)
	prec, rec = dc.get_precision_recall(true_positions, tp_pred, diameters=diameters, threshold=0.5)
	print('tp', prec, rec)
	prec, rec = dc.get_precision_recall(true_positions, coords, diameters=diameters, threshold=0.5)
	print('local_max', prec, rec)

	ap, precisions, recalls, thresholds = dc.average_precision(true_positions, coords, diameters)

	fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='Unet', tag='o-', color='red')
	plt.show()

	x,y = dc.get_gr(true_positions, 50, 50, )
	plt.plot(x, y, label=f'true n ={len(true_positions)}', color='black')
	x,y = dc.get_gr(tp_pred, 50, 50, )
	plt.plot(x, y, label=f'trackpy n ={len(tp_pred)}', color='grey')
	x,y = dc.get_gr(coords, 50, 50, )
	plt.plot(x, y, label=f'local_max n ={len(coords)}', color='red')
	plt.legend()
	plt.show()


	dc.view(image, label=label, positions=coords)
