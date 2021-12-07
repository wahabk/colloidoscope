from colloidoscope.deepcolloid import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)



	canvas_size=(32,128,128)
	dataset_name = 'test'
	n_samples = 500
	# nums = dc.get_hdf5_keys('test')
	# print(nums)
	# exit()


	for n in range(78,n_samples+1):
		print(f'{n}/{n_samples}')
		zoom = 0.75
		xykernel = randrange(1,4,2)
		
		gauss = (randrange(5,8,2),xykernel,xykernel)
		gauss = (5,1,1)
		brightness = randrange(180,250)
		noise = uniform(0.002, 0.008)
		volfrac = 0.2 #randrange(2, 5)/10 
		canvas, positions, label = dc.simulate(canvas_size, zoom, gauss, noise=noise, volfrac = volfrac, debug=True)
		
		# dc.view(canvas, positions)
		# napari.run()
		print(canvas.shape, canvas.max(), canvas.min())
		print(label.shape, label.max(), label.min())

		projection = np.max(canvas, axis=0)
		plt.imsave('output/test_sim.png', projection, cmap='gray')
		projection = np.max(label, axis=0)
		plt.imsave('output/test_sim_label.png', projection, cmap='gray')
		exit()

		# dc.write_hdf5(dataset_name, n, canvas, positions)
		# dc.write_hdf5(dataset_name+'_labels', n, label)
		# canvas, positions, label = None, None, None
		