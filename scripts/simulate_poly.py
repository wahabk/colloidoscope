from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import numpy as np
import random

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(32,128,128)
	dataset_name = 'janpoly'
	n_samples_per_phi = 100 #5000
	num_workers = 10

	# keys = dc.get_hdf5_keys(dataset_name)
	# print(keys)
	# exit()
	
	# make 100 scans of each volfrac
	# make list of n_samples for each volfrac
	volfracs = [[round(v, 2)]*n_samples_per_phi for v in  np.linspace(0.1,0.5,5)]*2
	# more =  [[round(v, 2)]*n_samples_per_phi for v in  np.linspace(0.1,0.5,5)]
	volfracs = np.array(volfracs)
	print(volfracs.shape)


	index = 1
	for i, volfrac in enumerate(volfracs):
		i+=1
		for n, v in enumerate(volfrac):
			
			n+=1
			print(i, n, v, n+(i*100)-100, index)
			print('\n', f'{index}/{len(volfracs.flatten())}', '\n')
			
			# index+=1
			# continue

			volfrac = v
			types = {
			'very small' 	: {'r' : randrange(4,5), 'xy_gauss' : randrange(0,2), 'z_gauss' : randrange(1,3), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.02)},
			'small' 		: {'r' : randrange(5,8), 'xy_gauss' : randrange(0,3), 'z_gauss' : randrange(2,4), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.03)},
			'medium' 		: {'r' : randrange(8,11), 'xy_gauss' : randrange(0,4), 'z_gauss' : randrange(3,7), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.04)},
			'large' 		: {'r' : randrange(10,12), 'xy_gauss' : randrange(0,5), 'z_gauss' : randrange(5,8), 'min_brightness' : randrange(80,150), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.04)},
			}
			keys = list(types)
			this_type = random.choice(keys)

			r = types[this_type]['r']
			xy_gauss = types[this_type]['xy_gauss']
			z_gauss = types[this_type]['z_gauss']
			min_brightness = types[this_type]['min_brightness']
			max_brightness = types[this_type]['max_brightness']
			noise = types[this_type]['noise']

			metadata = {
				'dataset': dataset_name,
				'n' 	 : n,
				'type'	 : this_type,
				'volfrac': volfrac,
				'params' : types[this_type],
			}
			print(metadata)

			# hoomd_positions = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size)
			# path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
			if index > 500:
				path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
			else:
				path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
			print(f'Reading: {path} ...')
			hoomd_positions, diameters = read_gsd(path, n)
			print(diameters.shape)
			centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)

			metadata['n_particles'] = len(centers)
			canvas, label = dc.simulate(canvas_size, centers, r, xy_gauss, z_gauss, min_brightness, max_brightness, 
										noise, make_label=True, diameters=diameters, num_workers=num_workers)

			print(canvas.shape, canvas.max(), canvas.min())
			print(label.shape, label.max(), label.min())

			# dc.view(canvas, centers,  label)

			# exit()

			# projection = np.max(canvas, axis=0)
			# projection_label = np.max(label, axis=0)*255
			# sidebyside = np.concatenate((projection, projection_label), axis=1)
			# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

			dc.write_hdf5(dataset_name, n, canvas, metadata=metadata, positions=centers, label=label, diameters=diameters, dtype='uint8')
			# dc.write_hdf5(dataset_name+'_labels', n, label, metadata=None, positions=centers, dtype='float32')

			# canvas, metadata, positions, diameters = dc.read_hdf5(dataset_name, n, read_metadata=True, read_diameters=True)
			# print(canvas.shape, positions.shape, diameters.shape)
			# canvas, centers, label = None, None, None

			index+=1