from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform
import numpy as np
import random

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(32,128,128)
	dataset_name = 'jan22'
	n_samples = 500 #5000

	# keys = dc.get_hdf5_keys(dataset_name)
	# print(keys)
	# exit()
	
	# TODO change this to know which sim came from which pos
	# make list of volfracs already simulated
	volfracs= [round(x, 2) for x in np.linspace(0.1,0.55,10)]
	# make list of n_samples for each volfrac
	volfracs = np.array([[x]*n_samples for x in volfracs]).flatten()

	print(volfracs)
	for n, volfrac in enumerate(volfracs):
		
		n+=1
		print('\n', f'{n}/{len(volfracs)}', '\n')

		volfrac = random.choice(volfracs)
		types = {
		'very small' 	: {'r' : randrange(4,5), 'xy_gauss' : randrange(0,2), 'z_gauss' : randrange(1,3), 'min_brightness' : randrange(180,220), 'max_brightness' : randrange(221,255), 'noise': uniform(0, 0.01)},
		'small' 		: {'r' : randrange(5,8), 'xy_gauss' : randrange(0,3), 'z_gauss' : randrange(2,4), 'min_brightness' : randrange(150,200), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.02)},
		'medium' 		: {'r' : randrange(8,11), 'xy_gauss' : randrange(0,4), 'z_gauss' : randrange(3,7), 'min_brightness' : randrange(150,200), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.03)},
		'large' 		: {'r' : randrange(10,12), 'xy_gauss' : randrange(0,5), 'z_gauss' : randrange(5,8), 'min_brightness' : randrange(150,200), 'max_brightness' : randrange(201,255), 'noise': uniform(0, 0.03)},
		}
		keys = list(types)
		this_type = random.choice(keys)

		r = types[this_type]['r']
		xy_gauss = types[this_type]['xy_gauss']
		z_gauss = types[this_type]['z_gauss']
		brightness = types[this_type]['brightness']
		noise = types[this_type]['noise']

		metadata = {
			'dataset': dataset_name,
			'n' 	 : n,
			'type'	 : this_type,
			'volfrac': volfrac,
			'params' : types[this_type],
		}
		print(metadata)


		# TODO fix crop size
		# hoomd_positions = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size)
		path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
		hoomd_positions, diameters = read_gsd(path, randrange(0,500))
		centers = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2)
		metadata['n_particles'] = len(centers)
		canvas, label = dc.simulate(canvas_size, centers, r, xy_gauss, z_gauss, brightness, noise, make_label=True, diameters=diameters, num_workers=10)

		print(canvas.shape, canvas.max(), canvas.min())
		print(label.shape, label.max(), label.min())

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

		# TODO SAVE DIAMETERS PLEASE
		# TODO sort out indices of which one to sim

		dc.write_hdf5(dataset_name, n, canvas, metadata=metadata, positions=centers, dtype='uint8')
		dc.write_hdf5(dataset_name+'_labels', n, label, metadata=None, positions=centers, dtype='float32')

		canvas, metadata, positions = dc.read_hdf5(dataset_name, n)
		print(canvas.shape, positions.shape)

		canvas, centers, label = None, None, None
		