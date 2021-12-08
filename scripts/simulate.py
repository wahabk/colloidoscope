from colloidoscope.hoomd_sim_positions import convert_hoomd_positions, hooomd_sim_positions
from colloidoscope import DeepColloid
from colloidoscope.simulator import simulateimport numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import numpy as np

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)



	canvas_size=(32,128,128)
	dataset_name = 'test'
	n_samples = 10
	# nums = dc.get_hdf5_keys('test')
	# print(nums)
	# exit()

	types = {
		'very small' 	: {'r' : 3-4, 'xy_gauss' : 1, 'z_gauss' : 2, 'brightness' : 255, 'noise': 0.01},
		'small' 		: {'r' : 5-7, 'xy_gauss' : 2, 'z_gauss' : 3, 'brightness' : 150, 'noise': 0.02},
		'medium' 		: {'r' : 8-10, 'xy_gauss' : 3, 'z_gauss' : 6, 'brightness' : 150, 'noise': 0.03},
		'large' 		: {'r' : 11-12, 'xy_gauss' : 4, 'z_gauss' : 7, 'brightness' : 150, 'noise': 0.03},
	}



	for n in range(0,n_samples+1):
		print(f'{n}/{n_samples}')
		zoom = 0.75
		xykernel = randrange(1,4,2)
		
		gauss = (randrange(5,8,2),xykernel,xykernel)
		brightness = randrange(180,250)
		noise = uniform(0.002, 0.008)
		volfrac = randrange(2, 5)/10 

		this_type = 'very small'

		r = types[this_type]['r']
		xy_gauss = types[this_type]['xy_gauss']
		z_gauss = types[this_type]['z_gauss']
		brightness = types[this_type]['brightness']
		noise = types[this_type]['noise']


		centers = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size, diameter=10)
		canvas, positions, label = dc.simulate(canvas_size, centers, r, zoom, xy_gauss, z_gauss, brightness, noise)
		# dc.view(canvas, positions)
		# napari.run()
		print(canvas.shape, canvas.max(), canvas.min())
		print(label.shape, label.max(), label.min())

		projection = np.max(canvas, axis=0)
		projection = np.max(label, axis=0)
		plt.imsave('output/test_sim.png', projection, cmap='gray')
		plt.imsave('output/test_sim_label.png', projection, cmap='gray')
		exit()

		# dc.write_hdf5(dataset_name, n, canvas, positions)
		# dc.write_hdf5(dataset_name+'_labels', n, label)
		# canvas, positions, label = None, None, None
		