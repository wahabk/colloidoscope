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

	canvas_size=(100,100,100)
	label_size=(100,100,100)
	
	dataset_name = 'heatmap_1400'
	num_workers = 16
	heatmap_r = 'radius'
	n_samples_per_volfrac = 200
	n_per_type = 100 # for testing


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
			'very small' 	: {'r' : randrange(4,6), 	'particle_size' : uniform(0.01,1), 'cnr' : uniform(2, 10),  'brightness' : random.randrange(30, 200), 'snr' : uniform(2,10)},
			'medium' 		: {'r' : randrange(7,8), 	'particle_size' : uniform(0.01,1), 'cnr' : uniform(2, 10),  'brightness' : random.randrange(30, 200), 'snr' : uniform(2,10)},
			'large' 		: {'r' : randrange(8,14), 	'particle_size' : uniform(0.01,1), 'cnr' : uniform(2, 10),  'brightness' : random.randrange(30, 200), 'snr' : uniform(2,10)},
			}

			keys = list(types.keys())
			this_type = random.choice(keys)
			params = types[this_type]
			params['f_sigma'] = 10
			params['b_sigma'] = 15

			metadata = {
				'dataset': dataset_name,
				'n' 	 : index,
				'type'	 : this_type,
				'volfrac': volfrac,
				'params' : params,
			}

			path = f'{dataset_path}/Positions/big/phi{volfrac*1000:.0f}.gsd'
			print(f'Reading: {path} at {n+1} ...')

			hoomd_positions, diameters = read_gsd(path, n+1)

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=True, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 
			final_diameters = final_diameters*params['r']*2

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

	canvas_size=(200,200,200)
	label_size=(200,200,200)

	params = dict(
		r=10,
		particle_size=0.25,
		snr=3,
		cnr=3,
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

	path = f"{dataset_path}/Positions/old/phi{params['volfrac']*1000:.0f}.gsd"
	hoomd_positions, diameters = read_gsd(path, 499)

	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()

	canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
								params['snr'], diameters=diameters, make_label=True, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
	metadata['n_particles'] = len(final_centers) # this might depend on label size 

	final_diameters = final_diameters*params['r']*2
	dc.write_hdf5(test_dataset_name, 10000, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')


	# Sim for testing
	canvas_size=(100,100,100)
	label_size=(100,100,100)

	test_params = ['volfrac','particle_size','r','brightness','cnr','snr',]
	test_list = [[s]*n_per_type for s in test_params]
	
	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()

	index=1 # 0 index for gr
	for i, t in enumerate(test_list):
		for n, this_type in enumerate(t):
			gsd_index = n

			index = (n_per_type*i)+n

			types = {
				'r' 			: {'r' : randrange(4,14), 	'particle_size' : 1, 				'cnr' : 10,							'brightness' : 255, 						'snr' : 10, 						'volfrac' : 0.2},
				'particle_size' : {'r' : 10, 				'particle_size' : uniform(0.1,1), 	'cnr' : 10,							'brightness' : 255, 						'snr' : 10, 						'volfrac' : 0.2},
				'cnr' 			: {'r' : 10, 				'particle_size' : 1, 				'cnr' : uniform(0.1, 5),			'brightness' : 255, 						'snr' : 10, 						'volfrac' : 0.2},
				'brightness' 	: {'r' : 10, 				'particle_size' : 1, 				'cnr' : 10,							'brightness' : randrange(10, 255), 			'snr' : 10, 						'volfrac' : 0.2},
				'snr' 			: {'r' : 10, 				'particle_size' : 1, 				'cnr' : 10,							'brightness' : 255, 						'snr' : uniform(0.1,5), 			'volfrac' : 0.2},			
				'volfrac' 		: {'r' : 10, 				'particle_size' : 1, 				'cnr' : 10,							'brightness' : 255, 						'snr' : 10, 						'volfrac' : random.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])},
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
			volfrac = params['volfrac']

			params['f_sigma'] = 10
			params['b_sigma'] = 5

			metadata = {
				'dataset': test_dataset_name,
				'n' 	 : index,
				'type'	 : this_type,
				'volfrac': volfrac,
				'params' : params,
			}

			path = f'{dataset_path}/Positions/old/phi{volfrac*1000:.0f}.gsd'
			print(f'Reading: {path} at {gsd_index} ...')

			hoomd_positions, diameters = read_gsd(path, gsd_index)

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=True, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 
			final_diameters = final_diameters*r*2
			print(metadata)

			dc.write_hdf5(test_dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
			index+=1

	keys = dc.get_hdf5_keys(test_dataset_name)
	print(keys)
