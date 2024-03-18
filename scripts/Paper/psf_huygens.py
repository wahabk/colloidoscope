from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import numpy as np
import random
import psf
from scipy import ndimage
from pathlib2 import Path
import psf

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)


	# particle_size = 5
	# args = dict(shape=(64, 64), dims=(particle_size, particle_size), ex_wavelen=488, em_wavelen=520, num_aperture=1.2, refr_index=1.4, pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
	# obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
	# psf_kernel = obsvol.volume()
	# psf_kernel = ndimage.zoom(psf_kernel, 0.25)
	# print(psf_kernel.shape)
	# dc.view(psf_kernel)

	real_psfs = Path(dataset_path) / 'Real/PSF/'

	paths = real_psfs.iterdir()

	for p in paths:
		print(p.stem)
		psf = dc.read_tif(str(p))
		print(psf.shape)
		dc.view(psf)

	


