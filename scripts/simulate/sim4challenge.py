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
import h5py

def write_sim_dataset(path, n, canvas, positions, NO_DIAMETERS, metadata):
    with h5py.File(path, "a") as f:
        dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype=dtype, data = canvas, compression=1)
    
    # TODO write positions and metadata as pandas csv
    # TODO remove diameters from read.py and metric.py
    # TODO make x_train, y_train, x_test, y_test


    return

if __name__ == '__main__':
	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(100,100,100)
	label_size=(100,100,100)
	
	dataset_name = 'gauss_1400'
	num_workers = 16
	# heatmap_r = 'radius'
	n_samples_per_volfrac = 200

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