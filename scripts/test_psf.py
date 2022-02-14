from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import numpy as np
import random
import psf
from scipy.signal import convolve
from scipy.ndimage import zoom

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(32,128,128)
	dataset_name = 'feb_blur'
	n_samples_per_phi = 99 #5000
	num_workers = 10

	data = dc.read_hdf5(dataset_name, 1)
	image = data['image']

	r = 5
	xy_gauss = 1
	z_gauss = 1
	max_brightness = 255
	min_brightness = 200
	noise = 0

	volfrac = 0.4
	path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
	hoomd_positions, diameters = read_gsd(path, randrange(0,100))
	centers = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2)

	blank, label = dc.simulate(canvas_size, centers, r, xy_gauss, z_gauss, min_brightness, max_brightness, noise, make_label=True, diameters=diameters, num_workers=10)
	blank = zoom(blank, 2)

	args = dict(shape=(64, 64), dims=(4, 4), ex_wavelen=633, em_wavelen=643,
				num_aperture=1.2, refr_index=1.4,
				pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
	obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)

	array = obsvol.volume()

	print(array.shape)



	dc.view(np.max(array, axis=1))
	
	# import pdb; pdb.set_trace()
	
	dc.view(blank)

	blurred = convolve(blank, array, mode='same')

	print(blurred.shape)
	dc.view(blurred)



	# expsf = obsvol.expsf
	# empsf = obsvol.empsf
	# gauss = gauss2 = psf.PSF(psf.GAUSSIAN | psf.EXCITATION, **args)


