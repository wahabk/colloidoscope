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
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	
	dataset_name = 'psf_seg_1400'
	num_workers = 10
	heatmap_r = 'seg-1'
	psf_kernel = 'standard'
	
	# make 100 scans of each volfrac
	# make list of lists of n_samples for each volfrac
	# poly_phis = [[round(v, 2)]*90 for v in  np.linspace(0.2,0.5,4)]
	# poly_phis = np.array(poly_phis)
	# print(poly_phis.shape, poly_phis[0])
	
	phis = [[round(x, 2)]*200 for x in np.linspace(0.25,0.55,7)]
	phis = np.array(phis)
	# make list of n_samples for each volfrac
	print(phis.shape)

	index = 1
	for i, volfracs in enumerate(phis):
		for n, v in enumerate(volfracs):
			print('\n', n, f'{index}/{len(phis.flatten())}', '\n')

			volfrac = v

			# define types of particles in simulation
			types = {
			'very small' 	: {'r' : randrange(4,6), 	'particle_size' : uniform(0.1,1.5), 'cnr' : triangular(0.2, 10, 0.5),  'brightness' : random.randrange(30, 200), 'snr' : triangular(0.1,10,3)},
			'medium' 		: {'r' : randrange(7,8), 	'particle_size' : uniform(0.1,1.5), 'cnr' : triangular(0.2, 10, 0.5),  'brightness' : random.randrange(30, 200), 'snr' : triangular(0.1,10,3)},
			'large' 		: {'r' : randrange(8,14), 	'particle_size' : uniform(0.1,1.5), 'cnr' : triangular(0.2, 10, 0.5),  'brightness' : random.randrange(30, 200), 'snr' : triangular(0.1,10,3)},
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

			# read huygens psf
			psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
			psf_kernel = dc.read_tif(str(psf_path))
			psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
			psf_kernel = psf_kernel/psf_kernel.max()

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 

			print(metadata)	
			print(canvas.shape, canvas.max(), canvas.min())
			print(label.shape, label.max(), label.min())

			# dc.view(canvas, final_centers, label)
			# plot_with_side_view(canvas, f'output/figs/simulation/{index}.png')
			# projection = np.max(canvas, axis=0)
			# projection_label = np.max(label, axis=0)*255
			# sidebyside = np.concatenate((projection, projection_label), axis=1)
			# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

			dc.write_hdf5(dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
			index+=1

	test_dataset_name = dataset_name+'_test'

	n_per_type = 50
	test_params = ['r','particle_size','b','cnr','snr','v',]
	test_list = [s*n_per_type for s in test_params]
	# Sim for testing
	
	for i, t in enumerate(test_list):
		for n, this_type in enumerate(t):

			index = random.randint(1,499)

			volfrac = v
			types = {
				'r' 			: {'r' : randrange(4,14), 	'particle_size' : 400, 				'cnr' : 4,							'brightness' : 100, 						'snr' : 4, 						'v' : 0.3},
				'particle_size' : {'r' : 6, 				'particle_size' : uniform(0.1,1.5), 'cnr' : 4,							'brightness' : 100, 						'snr' : 4, 						'v' : 0.3},
				'b' 			: {'r' : 6, 				'particle_size' : 400, 				'cnr' : triangular(0.2, 10, 0.5),	'brightness' : 100, 						'snr' : 4, 						'v' : 0.3},
				'cnr' 			: {'r' : 6, 				'particle_size' : 400, 				'cnr' : 4,							'brightness' : random.randrange(30, 200), 	'snr' : 4, 						'v' : 0.3},
				'snr' 			: {'r' : 6, 				'particle_size' : 400, 				'cnr' : 4,							'brightness' : 100, 						'snr' : triangular(0.1,10,3), 	'v' : 0.3},			
				'v' 			: {'r' : 6, 				'particle_size' : 400, 				'cnr' : 4,							'brightness' : 100, 						'snr' : 4, 						'v' : random.choice([0.25,0.3, 0.35, 0.4, 0.45, 0.5, 0.55])},
			}

			# define types of particles in simulation

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

			# read huygens psf
			psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
			psf_kernel = dc.read_tif(str(psf_path))
			psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
			psf_kernel = psf_kernel/psf_kernel.max()

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 

			print(metadata)
			print(canvas.shape, canvas.max(), canvas.min())
			print(label.shape, label.max(), label.min())

			# dc.view(canvas, final_centers, label)
			# plot_with_side_view(canvas, f'output/figs/simulation/{index}.png')
			# projection = np.max(canvas, axis=0)
			# projection_label = np.max(label, axis=0)*255
			# sidebyside = np.concatenate((projection, projection_label), axis=1)
			# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

			dc.write_hdf5(test_dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
			index+=1
