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

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	dataset_name = 'feb_final'
	num_workers = 10
	
	# make 100 scans of each volfrac
	# make list of lists of n_samples for each volfrac
	poly_volfracs = [[round(v, 2)]*90 for v in  np.linspace(0.1,0.5,5)]
	poly_volfracs = np.array(poly_volfracs)
	print(poly_volfracs.shape, poly_volfracs[0])
	
	volfracs= [round(x, 2) for x in np.linspace(0.1,0.55,10)]
	# make list of n_samples for each volfrac
	volfracs = np.array([[x]*400 for x in volfracs])
	print(volfracs.shape, volfracs[0])


	# exit()

	index = 1
	for l in [poly_volfracs, volfracs]:
		for i, volfracs in enumerate(l):
			for n, v in enumerate(volfracs):			
				print('\n', n, f'{index}/{4450}', '\n')
				if index < 812 : 
					index +=1
					continue

				volfrac = v
				types = {
				'very small' 	: {'r' : randrange(4,5), 'xy_gauss' : randrange(0,2), 'z_gauss' : randrange(1,6), 'min_brightness' : randrange(50,150), 'max_brightness' : randrange(155,250), 'noise': uniform(0, 0.02)},
				'medium' 		: {'r' : randrange(5,8), 'xy_gauss' : randrange(0,3), 'z_gauss' : randrange(5,10), 'min_brightness' : randrange(50,150), 'max_brightness' : randrange(155,250), 'noise': uniform(0, 0.03)},
				'large' 		: {'r' : randrange(8,10), 'xy_gauss' : randrange(1,5), 'z_gauss' : randrange(8,11), 'min_brightness' : randrange(50,150), 'max_brightness' : randrange(155,250), 'noise': uniform(0, 0.04)},
				}
				keys = list(types.keys())
				this_type = random.choice(keys)

				r = types[this_type]['r']
				xy_gauss = types[this_type]['xy_gauss']
				z_gauss = types[this_type]['z_gauss']
				min_brightness = types[this_type]['min_brightness']
				max_brightness = types[this_type]['max_brightness']
				noise = types[this_type]['noise']

				metadata = {
					'dataset': dataset_name,
					'n' 	 : index,
					'type'	 : this_type,
					'volfrac': volfrac,
					'params' : types[this_type],
				}
				print(metadata)

				args = dict(shape=(64, 64), dims=(4, 4), ex_wavelen=488, em_wavelen=520,
							num_aperture=1.2, refr_index=1.4,
							pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
				obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
				kernel = obsvol.volume()

				kernel = ndimage.zoom(kernel, random.choice([0.1,0.2,0.3,0.4,0.5]))

				if index > 450:
					path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
					print(f'Reading: {path} at {n+1} ...')
				else:
					path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
					print(f'Reading: {path} at {n+1} ...')
				hoomd_positions, diameters = read_gsd(path, n+1)
								
				print(diameters.shape)
				centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)

				metadata['n_particles'] = len(centers)
				canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, centers, r, kernel, min_brightness, max_brightness,
											noise, make_label=True, diameters=diameters, num_workers=num_workers)

				print(canvas.shape, canvas.max(), canvas.min())
				print(label.shape, label.max(), label.min())

				# dc.view(canvas, final_centers, label)
				# projection = np.max(canvas, axis=0)
				# projection_label = np.max(label, axis=0)*255
				# sidebyside = np.concatenate((projection, projection_label), axis=1)
				# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')

				dc.write_hdf5(dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')
				index+=1