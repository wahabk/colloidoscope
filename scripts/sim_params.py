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

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	
	dataset_name = 'test'
	num_workers = 10
	heatmap_r = 5#'radius'
	index = 1
	volfrac = 0.5

	r = 16
	particle_size = 1.5
	cnr = 5
	f_mean = 200
	snr = 10

	path = f'{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd'
	hoomd_positions, diameters = read_gsd(path, index)

	canvas, label, final_centers, final_diameters = dc.simulate(canvas_size, hoomd_positions, r, particle_size, f_mean, cnr,
								snr, diameters=diameters, make_label=True, label_size=label_size, heatmap_r=heatmap_r, num_workers=num_workers)


	dc.view(canvas, final_centers, label)
	# plot_with_side_view(canvas, f'output/figs/simulation/{index}.png')
	# projection = np.max(canvas, axis=0)
	# projection_label = np.max(label, axis=0)*255
	# sidebyside = np.concatenate((projection, projection_label), axis=1)
	# plt.imsave('output/test_sim.png', sidebyside, cmap='gray')


	estimated_noise = np.array([dc.estimate_noise(s) for s in canvas]).mean()
	foreground = canvas[label > 0.001]
	f_m = foreground.mean()
	estimated_snr = f_mean / estimated_noise
	# noise = (255 / snr) / (f_mean*10)
	noise = (f_mean / (snr * 255))/10

	#TODO sweeep snr?

	print(f"real brightness {f_mean}, estimated {f_m}")
	print(f"requested noise {noise} or {noise*255}, measured {estimated_noise}")
	print(f"requested snr {snr}, measured {estimated_snr}")

