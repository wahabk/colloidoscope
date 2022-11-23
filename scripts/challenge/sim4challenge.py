from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import random
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path
import h5py
import pandas as pd

def write_sim_dataset(dataset_path, index, canvas, positions, metadata):
	x_path =  dataset_path+"/x_train.hdf5"
	with h5py.File(x_path, "a") as f:
		dset = f.create_dataset(name=str(index), shape=canvas.shape, dtype="uint8", data = canvas, compression=1)
	
	y_path =  dataset_path+"/y_train.csv"
	if Path(y_path).is_file():
		og = pd.read_csv(y_path, index_col="index")
		indices = np.full((len(positions),1), fill_value=index)
		positions_with_indices = np.concatenate([indices, positions], axis=1)
		pos_df = pd.DataFrame(positions_with_indices)
		pos_df.columns = ["index", "Z", "X", "Y"]
		pos_df = pos_df.set_index("index")
		new_og = pd.concat([og, pos_df], axis=0, ignore_index=False)
		new_og.to_csv(y_path)
	else:
		indices = np.full((len(positions),1), fill_value=index)
		positions_with_indices = np.concatenate([indices, positions], axis=1)
		pos_df = pd.DataFrame(positions_with_indices)
		pos_df.columns = ["index", "Z", "X", "Y"]
		pos_df = pos_df.set_index("index")
		pos_df.to_csv(y_path)

	meta_path = dataset_path+"/x_train_metadata.csv"
	if Path(meta_path).is_file():
		og = pd.read_csv(meta_path, index_col=0)
		d = {index : metadata}
		meta_df = pd.DataFrame.from_dict(d, orient='index')
		new_og = pd.concat([og, meta_df], axis=0, ignore_index=False)
		print(new_og)

		new_og.to_csv(meta_path)
	else:
		d = {index : metadata}
		meta_df = pd.DataFrame.from_dict(d, orient='index')
		meta_df.to_csv(meta_path)
		
	return

if __name__ == '__main__':
	dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	
	dataset_name = 'x_train'
	out_path = '/mnt/scratch/ak18001/Colloids/'
	num_workers = 16
	heatmap_r = 'radius'
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

	index = 0
	for i, volfracs in enumerate(phis):
		for n, v in enumerate(volfracs):
			print('\n', n, f'{index}/{len(phis.flatten())}', '\n')

			volfrac = v

			# define types of particles in simulation
			types = {
			'very small' 	: {'r' : randrange(4,6), 	'particle_size' : uniform(0.1,1), 'cnr' : uniform(1, 10),  'brightness' : random.randrange(30, 200), 'snr' : uniform(1,10)},
			'medium' 		: {'r' : randrange(7,8), 	'particle_size' : uniform(0.1,1), 'cnr' : uniform(1, 10),  'brightness' : random.randrange(30, 200), 'snr' : uniform(1,10)},
			'large' 		: {'r' : randrange(8,14), 	'particle_size' : uniform(0.1,1), 'cnr' : uniform(1, 10),  'brightness' : random.randrange(30, 200), 'snr' : uniform(1,10)},
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
				**params,
			}

			path = f'{dataset_path}/Positions/old/phi{volfrac*1000:.0f}.gsd'
			print(f'Reading: {path} at {n+1} ...')

			hoomd_positions, diameters = read_gsd(path, n+1)

			canvas, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, params['r'], params['particle_size'], params['brightness'], params['cnr'],
										params['snr'], diameters=diameters, make_label=False, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)
			metadata['n_particles'] = len(final_centers) # this might depend on label size 
			final_diameters = final_diameters*params['r']*2

			print(metadata)
			print(canvas.shape, canvas.max(), canvas.min())
			# print(label.shape, label.max(), label.min())

			# code for debugging
			# dc.view(canvas, final_centers, label)
			# plot_with_side_view(canvas, f'output/figs/simulation/{index}.png')
			# projection = np.max(canvas, axis=0)
			# projection_label = np.max(label, axis=0)*255
			# sidebyside = np.concatenate((projection, projection_label), axis=1)
			# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

			write_sim_dataset(dataset_path, index, canvas=canvas, positions=final_centers, metadata=metadata)
			index+=1