from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
import numpy as np
import math
from scipy.signal import convolve2d
from pathlib2 import Path
from scipy import ndimage

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	
	dataset_name = 'test_heatmap'
	num_workers = 12
	heatmap_r = 'seg-1'
	index = 0
	volfrac = 0.55

	r = 5
	particle_size = 0.2
	cnr = 3
	f_mean = 100
	snr = 4

	params = {
		'r' : r,
		'particle_size' : particle_size,
		'cnr' : cnr,
		'brightness' : f_mean,
		'f_sigma' : 30,
		'b_sigma' : 20,
	}

	path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
	hoomd_positions, diameters = read_gsd(path, index)

	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()
	# print(psf_kernel.shape, psf_kernel.max(), psf_kernel.dtype)
	# dc.view(psf_kernel)

	# psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedZ.tif'
	# psf_kernel = dc.read_tif(str(psf_path))
	# psf_kernel = dc.crop3d(psf_kernel, (48,16,16), (24,159,160))
	# print(psf_kernel.shape, psf_kernel.max(), psf_kernel.dtype)
	# psf_kernel = psf_kernel/psf_kernel.max()
	# psf_kernel = ndimage.zoom(psf_kernel, 1)
	# dc.view(psf_kernel)

	# psf_kernel = 'standard'

	canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, r, particle_size, f_mean, cnr,
								snr, diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers, psf_kernel=psf_kernel)

	metadata = {
		'dataset': dataset_name,
		'n' 	 : index,
		'type'	 : 'very_small',
		'volfrac': volfrac,
		'params' : params,
	}

	dc.view(canvas, final_centers, label)
	# plot_with_side_view(canvas, f'output/figs/simulation/{index}.png')
	# projection = np.max(canvas, axis=0)
	# projection_label = np.max(label, axis=0)*255
	# sidebyside = np.concatenate((projection, projection_label), axis=1)
	# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')


	# estimated_noise = np.array([dc.estimate_noise(s) for s in canvas]).mean()
	# foreground = canvas[label > 0.001]
	# f_m = foreground.mean()
	# estimated_snr = f_mean / estimated_noise
	# # noise = (255 / snr) / (f_mean*10)
	# noise = (f_mean / (snr * 255))/10
	# print(f"real brightness {f_mean}, estimated {f_m}")
	# print(f"requested noise {noise} or {noise*255}, measured {estimated_noise}")
	# print(f"requested snr {snr}, measured {estimated_snr}")

	# dc.write_hdf5(dataset_name, index, canvas, metadata=metadata, positions=final_centers, label=label, diameters=final_diameters, dtype='uint8')