from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
# from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import numpy as np
import random
import psf
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path
from numba import njit

from scipy.spatial.distance import pdist, cdist

from tqdm import tqdm


from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max








if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(60,60,60)


	r = 5
	d = r*2
	index = 2
	volfrac = 0.1


	dataset_name = 'new_1400_30nm'

	d = dc.read_hdf5(dataset_name, index)
	image = d['image']
	label = d["label"]
	# positions = d["positions"]
	# metadata = d["metadata"]
	# diameters = d["diameters"]

	metadata, positions, diameters = dc.read_metadata(dataset_name, index)

	# label = dc.crop3d(label, label_size)

	tp_pred, df = dc.run_trackpy(image, diameter = dc.round_up_to_odd(metadata['params']['r']*2), )

	# positions, diameters = crop_positions_for_label(positions, canvas_size, label_size, diameters)

	prec, rec = dc.get_precision_recall(positions, tp_pred, diameters=diameters, threshold=0.5)
	print(prec, rec)

	# print(label.dtype)
	# print(label.shape, label.max(), label.min())

	distance = ndi.distance_transform_edt(np.array(label*255, dtype="uint16"))
	coords = peak_local_max(label)
	mask = np.zeros(distance.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers, _ = ndi.label(mask)
	new_label = watershed(-distance, markers, mask=label)

	print(distance.shape, distance.max(), distance.min())
	# print(coords.shape, coords.max(), coords.min())
	print(np.shape(coords))
	print(mask.shape, mask.max(), mask.min())
	print(new_label.shape, new_label.max(), new_label.min())

	# import pdb; pdb.set_trace()

	coords = np.array(coords)
	prec, rec = dc.get_precision_recall(positions, coords, diameters=diameters, threshold=0.5)
	print(prec, rec)


	dc.view(image, label=new_label)
