from colloidoscope import DeepColloid
import matplotlib.pyplot as plt
import numpy as np
import random
from tifffile import imsave
from scipy import ndimage
from skimage import io
from scipy.signal import convolve, convolve2d
import math 

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	return sidebyside


def read_real_examples():

	d = {}

	d["A - Silica (560nm 0.55Φ)"] = {}
	d["A - Silica (560nm 0.55Φ)"]['diameter'] = [17,15,15]
	d["A - Silica (560nm 0.55Φ)"]['volfrac'] = 0.55
	d["A - Silica (560nm 0.55Φ)"]['array'] = io.imread('examples/Data/james.tiff')
	d["B - Silica Decon (560nm 0.55Φ)"] = {}
	d["B - Silica Decon (560nm 0.55Φ)"]['diameter'] = [17,15,15]
	d["B - Silica Decon (560nm 0.55Φ)"]['volfrac'] = 0.55
	d["B - Silica Decon (560nm 0.55Φ)"]['array'] = io.imread('examples/Data/jamesdecon.tiff')
	# d["E - Silica (500nm 0.50Φ) "] = {}
	# d["E - Silica (500nm 0.50Φ) "]['diameter'] = 13
	# d["E - Silica (500nm 0.50Φ) "]['volfrac'] = 0.5
	# d["E - Silica (500nm 0.50Φ) "]['array'] = io.imread('examples/Data/emily.tiff')
	d["C - PMMA (315nm 0.58Φ)"] = {}
	d["C - PMMA (315nm 0.58Φ)"]['diameter'] = [15,11,11]
	d["C - PMMA (315nm 0.58Φ)"]['volfrac'] = 0.58
	d["C - PMMA (315nm 0.58Φ)"]['array'] = io.imread('examples/Data/levke.tiff')
	d["D - Emulsion (3μm 0.64Φ)"] = {}
	d["D - Emulsion (3μm 0.64Φ)"]['diameter'] = 15
	d["D - Emulsion (3μm 0.64Φ)"]['volfrac'] = 0.64
	array = io.imread('examples/Data/abraham.tiff') 
	d["D - Emulsion (3μm 0.64Φ)"]['array'] = ndimage.zoom(array, 2.25)
	d["E - Silica (1.2μm 0.2Φ)"] = {}
	d["E - Silica (1.2μm 0.2Φ)"]['diameter'] = 15
	d["E - Silica (1.2μm 0.2Φ)"]['volfrac'] = 0.2
	array  = io.imread('examples/Data/katherine.tiff')
	d["E - Silica (1.2μm 0.2Φ)"]['array'] = ndimage.zoom(array, 2)

	return d

def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

def signaltonoise2(array):
    signal = array.mean()
    noise = array.std()
    # noise = np.array([estimate_noise(s) for s in array]).mean()
    return signal/noise

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	real_dict = read_real_examples()
 
	real_len = len(real_dict)
	fig, axs = plt.subplots(1, real_len)
	plt.tight_layout()
 
	for i, (name, d) in enumerate(real_dict.items()):
		array = d['array']
		array = np.array((array/array.max())*255,dtype='uint8')
		print(name, array.mean(), array.min(), array.max())
		snr = signaltonoise2(array)

		array = array.flatten()
		# array.sort()

		axs[i].hist(array, bins=50)
		axs[i].set_xlabel('Brightness', fontsize=12)
		axs[i].set_ylabel('Frequency', fontsize=12)
		axs[i].set_xlim(0,255)
		axs[i].set_title(f'{name} SNR = {snr:.2f}', fontsize=10, rotation=7.5)
		# ax.legend(loc='upper left')
	fig.set_figwidth(12)
	fig.set_figheight(2)
	plt.savefig("output/figs/real_data/real_snr.png", bbox_inches="tight")



	