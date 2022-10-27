"""
The dataset for this project includes two files, an hdf5 file with scans and labels, and a json file with the metdata. 
The hdf5 file contains all 1000 64x64x64 scans (x’s), labels, true positions of particles, and their diameters. We usually train the model on creating semantic segmentation from the labels but your y will depend on what approach you take. The final output of the approach should be the true positions. (with or without model post processing)
To read this data we recommend using DeepColloid.read_hdf5()
The metadata parameters are as follows:
    • volfrac: volume fraction or density of spheres in the volume (usually between 0.1 and 0.5)
    • r: radius in pixels in the image
    • particle_size: we use this to define how small the particles would look through the microscope 
      (between 0.1 and 0.5 micrometers), this determines how bad the point spread function is in the simulation.
	  This combined with `r` provides the nm/pixel of the image
    • brightness: particle brightness between 80-255
    • SNR: signal to noise ratio
    • CNR: contrast to noise ratio
    • b_sigma and f_sigma, standard deviations of foreground and background noise (please read on the contrast to noise ratio equation)
"""

from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label
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



def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside, cmap='gray')
	plt.clf()

if __name__ == '__main__':
	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	
	dataset_name = 'sim_1000_radii'
	num_workers = 16
	heatmap_r = 'radius'
	n_samples_per_volfrac = 150
	n_per_type = 50 # for testing


	# psf_kernel = 'standard' #TODO make this change psf
	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()

	# each volfrac has 500, make 200 for training and validation and rest for testing
	phis = np.array([[round(x, 2)]*n_samples_per_volfrac for x in np.linspace(0.25,0.55,7)])
	print(phis.shape)

	index = 1
	for i, volfracs in enumerate(phis):
		for n, v in enumerate(volfracs):
			print('\n', n, f'{index}/{len(phis.flatten())}', '\n')

			volfrac = v

			# define types of particles in simulation
			types = {
			'very small' 	: {'r' : randrange(4,6), 	'particle_size' : uniform(0.1,1.5), 'cnr' : triangular(1, 10, 0.5),  'brightness' : random.randrange(30, 200), 'snr' : triangular(1,10,3)},
			'medium' 		: {'r' : randrange(7,8), 	'particle_size' : uniform(0.1,1.5), 'cnr' : triangular(1, 10, 0.5),  'brightness' : random.randrange(30, 200), 'snr' : triangular(1,10,3)},
			'large' 		: {'r' : randrange(8,14), 	'particle_size' : uniform(0.1,1.5), 'cnr' : triangular(1, 10, 0.5),  'brightness' : random.randrange(30, 200), 'snr' : triangular(1,10,3)},
			}

			keys = list(types.keys())
			this_type = random.choice(keys)
			params = types[this_type]
			params['f_sigma'] = 30
			params['b_sigma'] = 20

			metadata = {
				'dataset': dataset_name,
				'n' 	 : index,
				'type'	 : this_type,
				'volfrac': volfrac,
				'params' : params,
			}

			path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
			print(f'Reading: {path} at {n+1} ...')

			hoomd_positions, diameters = read_gsd(path, n+1)

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 
			

			print(metadata)
			print(canvas.shape, canvas.max(), canvas.min())
			print(label.shape, label.max(), label.min())

			# code for debugging
			# dc.view(canvas, final_centers, label)
			# plot_with_side_view(canvas, f'output/figs/simulation/{index}.png')
			# projection = np.max(canvas, axis=0)
			# projection_label = np.max(label, axis=0)*255
			# sidebyside = np.concatenate((projection, projection_label), axis=1)
			# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

			dc.write_hdf5(dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
			index+=1

	# sim 4 gr
	test_dataset_name = dataset_name+'_test'
	index=0

	canvas_size=(160,160,160)
	label_size=(160,160,160)

	params = dict(
		r=6,
		particle_size=0.3,
		snr=4,
		cnr=4,
		volfrac=0.55,
		brightness=100,
	)

	metadata = {
		'dataset': test_dataset_name,
		'n' 	 : index,
		'type'	 : '4gr',
		'volfrac': params['volfrac'],
		'params' : params,
	}

	path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
	hoomd_positions, diameters = read_gsd(path, 499)

	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()

	canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
								params['snr'], diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
	metadata['n_particles'] = len(final_centers) # this might depend on label size 

	dc.write_hdf5(test_dataset_name, 0, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')


	# Sim for testing
	canvas_size=(64,64,64)
	label_size=(64,64,64)

	test_params = ['r','particle_size','brightness','cnr','snr','v',]
	test_list = [[s]*n_per_type for s in test_params]
	
	index=1 # 0 index for gr
	for i, t in enumerate(test_list):
		for n, this_type in enumerate(t):

			gsd_index = (n_per_type*i)+n + n_samples_per_volfrac

			volfrac = v
			types = {
				'r' 			: {'r' : randrange(4,14), 	'particle_size' : 0.6, 				'cnr' : 8,							'brightness' : 200, 						'snr' : 8, 						'v' : 0.25},
				'particle_size' : {'r' : 8, 				'particle_size' : uniform(0.1,1.5), 'cnr' : 8,							'brightness' : 200, 						'snr' : 8, 						'v' : 0.3},
				'brightness' 	: {'r' : 8, 				'particle_size' : 0.6, 				'cnr' : triangular(0.1, 10, 3),		'brightness' : 200, 						'snr' : 8, 						'v' : 0.35},
				'cnr' 			: {'r' : 8, 				'particle_size' : 0.6, 				'cnr' : 8,							'brightness' : randrange(30, 200), 			'snr' : 8, 						'v' : 0.4},
				'snr' 			: {'r' : 8, 				'particle_size' : 0.6, 				'cnr' : 8,							'brightness' : 200, 						'snr' : triangular(0.1,10,3), 	'v' : 0.45},			
				'v' 			: {'r' : 8, 				'particle_size' : 0.6, 				'cnr' : 8,							'brightness' : 200, 						'snr' : 8, 						'v' : random.choice([0.25,0.3, 0.35, 0.4, 0.45, 0.5, 0.55])},
			}

			# define types of particles in simulation

			keys = list(types.keys())
			# this_type = random.choice(keys)
			params = types[this_type]
			r = params['r']
			particle_size = params['particle_size']
			brightness = params['brightness']
			cnr = params['cnr']
			snr = params['snr']
			volfrac = params['v']

			params['f_sigma'] = 30
			params['b_sigma'] = 20

			metadata = {
				'dataset': dataset_name,
				'n' 	 : gsd_index,
				'type'	 : this_type,
				'volfrac': volfrac,
				'params' : params,
			}

			path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
			print(f'Reading: {path} at {gsd_index+1} ...')

			hoomd_positions, diameters = read_gsd(path, gsd_index+1)

			# read huygens psf
			psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
			psf_kernel = dc.read_tif(str(psf_path))
			psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
			psf_kernel = psf_kernel/psf_kernel.max()

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 
			print(metadata)

			dc.write_hdf5(test_dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
			index+=1

	keys = dc.get_hdf5_keys(test_dataset_name)
	print(keys)
