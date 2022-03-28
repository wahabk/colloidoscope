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

	r = 10
	max_brightness = 255
	min_brightness = 100
	noise = 0

	args = dict(shape=(64, 64), dims=(4, 4), ex_wavelen=405, em_wavelen=415,
				num_aperture=1.2, refr_index=1.4,
				pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
	obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
	kernel = obsvol.volume()
	print(kernel.shape)
	kernel = zoom(kernel, random.choice([0.25,0.5,0.75,1]))
	print(kernel.shape)

	# dc.view(np.max(kernel, axis=1))

	volfrac = 0.4
	path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
	hoomd_positions, diameters = read_gsd(path, randrange(0,100))
	centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)

	print(centers.shape)

	canvas, label = dc.simulate(canvas_size, centers, r, kernel, min_brightness, max_brightness, noise, make_label=True, diameters=diameters, num_workers=10)
	print(canvas.shape)
	
	dc.view(canvas, positions=centers, label=label)



