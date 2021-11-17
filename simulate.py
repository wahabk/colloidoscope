from src.deepcolloid import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform


if __name__ == '__main__':
	dc = DeepColloid('/home/wahab/Data/HDD/Colloids')
	# dc = DeepColloid('/mnt/storage/home/ak18001/scratch/Colloids')

	canvas_size=(32,128,128)
	dataset = 'replicate'
	n_samples = 50

	for n in range(1,n_samples+1):
		print(f'{n}/{n_samples}')
		zoom = 0.75
		xykernel = randrange(1,4,2)
		
		gauss = (randrange(5,8,2),xykernel,xykernel)
		gauss = (5,1,1)
		brightness = randrange(180,250)
		noise = uniform(0.002, 0.008)
		canvas, positions, label = dc.simulate(canvas_size, zoom, gauss, noise=noise, volfrac = 0.3, debug=True)
		
		# dc.view(canvas, positions)
		# napari.run()

		dc.write_hdf5(dataset, n, canvas, positions)
		dc.write_hdf5(dataset+'_labels', n, label)
		canvas, positions, label = None, None, None
		