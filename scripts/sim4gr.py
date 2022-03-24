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

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside, cmap='gray')
	plt.clf()

def make_proj(array, label):

	array_projection = np.max(array, axis=0)
	label_projection = np.max(label, axis=0)*255
	# new_label = np.zeros_like(array_projection)

	if label_projection.shape != array_projection.shape:
		label_projection.resize(array_projection.shape)

	sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	sidebyside /= sidebyside.max()

	return sidebyside

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(160,160,160)
	label_size=(160,160,160)
	
	dataset_name = 'test'
	num_workers = 10

	index=1

	volfrac = 0.5 #random.choice([0.1,0.3,0.5])

	types = {
	'very small' 	: {'r' : randrange(4,5), 'psf_zoom' : random.choice([0.1,0.2,0.3,0.4,0.5]), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(155,250), 'noise': uniform(0, 0.02)},
	'medium' 		: {'r' : randrange(5,8), 'psf_zoom' : random.choice([0.1,0.2,0.3,0.4,0.5]), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(155,250), 'noise': uniform(0, 0.03)},
	'large' 		: {'r' : randrange(8,10), 'psf_zoom' : random.choice([0.1,0.2,0.3,0.4,0.5]), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(155,250), 'noise': uniform(0, 0.04)},
	}
	keys = list(types.keys())
	this_type = 'very small' # random.choice(keys)

	r = types[this_type]['r']
	min_brightness = types[this_type]['min_brightness']
	max_brightness = types[this_type]['max_brightness']
	noise = types[this_type]['noise']
	psf_zoom = types[this_type]['psf_zoom']

	metadata = {
		'dataset': dataset_name,
		'n' 	 : index,
		'type'	 : this_type,
		'volfrac': volfrac,
		'params' : types[this_type],
	}
	print(metadata)

	args = dict(shape=(64, 64), dims=(4, 4), ex_wavelen=488, em_wavelen=520,
				num_aperture=1.2, refr_index=1.4,
				pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
	obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
	kernel = obsvol.volume()

	kernel = ndimage.zoom(kernel, psf_zoom)

	path = f'{dataset_path}/Positions/test/phi{volfrac*1000:.0f}.gsd'
	hoomd_positions, diameters = read_gsd(path, 0)
	
	print(diameters.shape)
	centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)
	metadata['n_particles'] = len(centers)

	canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, centers, r, kernel, min_brightness, max_brightness,
								noise, make_label=True, label_size=label_size, diameters=diameters, num_workers=num_workers)

	print(canvas.shape, canvas.max(), canvas.min())
	print(label.shape, label.max(), label.min())

	# dc.view(canvas, final_centers, label)
	sidebyside = make_proj(canvas, label)
	plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

	# dc.write_hdf5(dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
