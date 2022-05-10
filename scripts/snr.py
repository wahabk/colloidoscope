from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
import random
from tifffile import imsave
from scipy import ndimage
from skimage import io

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	return sidebyside

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

def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	real_dict = read_real_examples()

	fig, axs = plt.subplots(2,2)

	print(axs.flatten())

	for (name, d), ax in zip(real_dict.items(), axs.flatten()):
		array = d['array']
		print(name, array.mean(), array.min(), array.max())
		snr = signaltonoise(array, axis=None)

		array = array.flatten()
		array.sort()

		ax.hist(array, bins=50)
		ax.set_xlabel('Brightness')
		ax.set_ylabel('Frequency')
		ax.set_xlim(0,255)
		ax.set_title(f'{name} snr = {snr:.2f}')
		# ax.legend(loc='upper left')

	plt.tight_layout()
	plt.show()



	