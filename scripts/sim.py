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

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	
	dataset_name = 'psf_cnr_radius'
	num_workers = 10
	heatmap_r = 'radius'
	
	# make 100 scans of each volfrac
	# make list of lists of n_samples for each volfrac
	# poly_phis = [[round(v, 2)]*90 for v in  np.linspace(0.2,0.5,4)]
	# poly_phis = np.array(poly_phis)
	# print(poly_phis.shape, poly_phis[0])
	
	phis = [[round(x, 2)]*499 for x in np.linspace(0.25,0.55,7)]
	phis = np.array(phis)
	# make list of n_samples for each volfrac
	print(phis.shape, phis[0])

	index = 1
	for i, volfracs in enumerate(phis):
		for n, v in enumerate(volfracs):
			print('\n', n, f'{index}/{len(phis.flatten())}', '\n')

			volfrac = v

			# define types of particles in simulation
			types = {
			'very small' 	: {'r' : randrange(4,6), 'particle_size' : uniform(0.1,2), 'cnr' : uniform(0.1, 3),  'brightness' : random.randrange(80, 200), 'noise': uniform(0, 0.02)},
			'medium' 		: {'r' : randrange(7,8), 'particle_size' : uniform(0.1,2), 'cnr' : uniform(0.1, 3),  'brightness' : random.randrange(80, 200), 'noise': uniform(0.01, 0.03)},
			'large' 		: {'r' : randrange(8,16), 'particle_size' : uniform(0.1,2), 'cnr' : uniform(0.1, 3),  'brightness' : random.randrange(80, 200), 'noise': uniform(0.01, 0.04)},
			}

			keys = list(types.keys())
			this_type = random.choice(keys)
			params = types[this_type]

			r = params['r']
			particle_size = params['particle_size']
			cnr = params['cnr']
			b = params['brightness']
			noise = params['noise']
			f_mean = params['brightness']

			# calculate foreground and background STD from Contrast to Noise Ratio equation
			f_sigma = 30
			b_sigma = 20
			b_mean = abs(cnr * b_sigma + f_sigma)
			print('cnr', f_mean,f_sigma,b_sigma,b_mean)

			params['f_sigma'] = f_sigma
			params['b_sigma'] = b_sigma
			params['b_mean'] = b_mean

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
			
			print(diameters.shape)
			centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)

			canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, centers, r, particle_size, f_mean, f_sigma, b_mean, b_sigma,
										noise, diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers)
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