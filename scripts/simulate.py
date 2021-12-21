from colloidoscope import hoomd_sim_positions
from colloidoscope.hoomd_sim_positions import convert_hoomd_positions, hooomd_sim_positions, read_gsd
from colloidoscope import DeepColloid
from colloidoscope.simulator import simulate
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
	dataset_name = 'third_run'
	n_samples = 1000
	# nums = dc.get_hdf5_keys('test')
	# print(nums)
	# exit()

	for n in range(122,n_samples+1):
		print('\n', f'{n}/{n_samples}', '\n')

		volfrac = round(random.choice(np.linspace(0.1,0.55,10)), 2)
		types = {
		'very small' 	: {'r' : randrange(3,5), 'xy_gauss' : randrange(0,2), 'z_gauss' : randrange(1,3), 'brightness' : 255, 'noise': uniform(0, 0.01)},
		'small' 		: {'r' : randrange(5,8), 'xy_gauss' : randrange(0,3), 'z_gauss' : randrange(2,4), 'brightness' : randrange(150,255), 'noise': uniform(0, 0.02)},
		'medium' 		: {'r' : randrange(8,11), 'xy_gauss' : randrange(0,4), 'z_gauss' : randrange(3,7), 'brightness' : randrange(150,255), 'noise': uniform(0, 0.03)},
		'large' 		: {'r' : randrange(10,12), 'xy_gauss' : randrange(0,5), 'z_gauss' : randrange(5,8), 'brightness' : randrange(150,255), 'noise': uniform(0, 0.03)},
		}
		keys = list(types)
		this_type = random.choice(keys)
		# this_type= 'small'

		print(this_type)
		print(types[this_type], volfrac)

		r = types[this_type]['r']
		xy_gauss = types[this_type]['xy_gauss']
		z_gauss = types[this_type]['z_gauss']
		brightness = types[this_type]['brightness']
		noise = types[this_type]['noise']

		# continue

		# hoomd_positions = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size)
		path = f'output/Positions/phi{volfrac*1000:.0f}.gsd'
		hoomd_positions, diameters = read_gsd(path, randrange(0,500))
		centers = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2)
		print(centers.shape)
		print(centers.min(), centers.max())
		canvas, label = dc.simulate(canvas_size, centers, r, xy_gauss, z_gauss, brightness, noise, make_label=True, num_workers=10)
		print(canvas.shape, canvas.max(), canvas.min())
		print(label.shape, label.max(), label.min())

		metadata = {
			'dataset': dataset_name,
			'n' 	 : n,
			'volfrac': volfrac,
			'params' : types[this_type],
		}

		# dc.view(canvas, centers)
		# viewer = napari.view_image(canvas, opacity=0.75)
		# viewer.add_image(label*255, opacity=0.75, colormap='red')
		# viewer.add_points(centers)
		# viewer.add_points(centers)
		# napari.run()

		# projection = np.max(canvas, axis=0)
		# projection_label = np.max(label, axis=0)*255
		# sidebyside = np.concatenate((projection, projection_label), axis=1)
		# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

		# dc.write_hdf5(dataset_name, n, canvas, centers, dtype='uint8')
		# dc.write_hdf5(dataset_name+'_labels', n, label, dtype='float32')
		canvas, centers, label = None, None, None
		