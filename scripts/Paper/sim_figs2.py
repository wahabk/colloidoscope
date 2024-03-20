from colloidoscope import DeepColloid
from colloidoscope.simulator import draw_spheres_sliced, crop_positions_for_label, make_background, crop3d
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
import random
import numpy as np
from skimage.util import random_noise
from scipy import ndimage
from scipy.signal import convolve
import matplotlib.pyplot as plt
from random import randrange, uniform, triangular
from pathlib2 import Path
import copy

from scripts.Paper.real_cnr import read_real_examples, GFP_CMAP

def plot_with_side_view(scan, path, cmap='gray'):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside, cmap=cmap)
	plt.clf()


def simulate(canvas_size:list, hoomd_positions:np.ndarray, r:int,
			particle_size:float, f_mean:float, cnr:float,
			snr:float, f_sigma:float=10, b_sigma:float=10, diameters=np.ndarray([]), make_label:bool=True, 
			heatmap_r='radius', num_workers=2, psf_kernel = 'standard'):
	'''
	psf between 0 and 1
	
	diameters, list of diameters from hoomdblue, where in default it is all ones, but in polydispersed data it is e.g. between 0.8-1.2
	
	particle size in um

	heatmap_r can be 'radius' to equal the particle or can be a constant int

	This will only work for a perfect cube eg 64x64x64 not cuboids

	psf_kernel = 'standard'

	'''
	centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)

	noise_std = (f_mean / (snr * f_mean))
	b_mean = f_mean - (cnr*noise_std*255)
	if b_mean < 0: b_mean=0

	brightnesses = np.array([random.gauss(f_mean, f_sigma) for _ in centers], dtype="float32")
	brightnesses[brightnesses>255]=255
	brightnesses=np.array(brightnesses, dtype="uint8")
	zoom = 0.5
	pad = 80
	gauss_kernel = (2, 2, 2)

	# zoom out to large image and positions
	# later we zoom back in to add aliasing
	zoom_out_size = [int(c/zoom) for c in canvas_size]
	# make bigger padded canvas
	# this is to allow the blur to work on the edges
	zoom_out_size_padded = [c+pad for c in zoom_out_size]
	canvas = make_background(zoom_out_size_padded, 2, 1, b_sigma/255, dtype='float32')*b_mean
	print(canvas.max(), canvas.min(), b_mean)
	canvas = np.array(canvas, dtype="uint8")
	print(canvas.max(), canvas.min())
	plot_with_side_view(canvas, f'output/Paper/sim_steps/1.png', cmap=GFP_CMAP)

	
	# convert centers to zoom out size
	zoom_out_centers=np.array(copy.deepcopy(centers))
	zoom_out_centers = zoom_out_centers / zoom
	zoom_out_centers = zoom_out_centers + (pad/2)


	diameters = np.array(diameters)
	# radii = [(d*r) for d in diameters]
	# zoom_out_radii = [(i/zoom) for i in radii]
	radii = diameters*r
	zoom_out_radii = radii/zoom

	if isinstance(psf_kernel, np.ndarray):
		nm_pixel = (particle_size*1000) / r 
		psf_nm_pixel = 50 # standard_huygens_pixel_nm
		psf_zoom = psf_nm_pixel / nm_pixel

		this_kernel = ndimage.zoom(psf_kernel, psf_zoom)

	else: raise ValueError(f"psf_kernel can be either str('Standard') or an np.ndarray but you provided {type(psf_kernel)}")

	# draw spheres slice by slice
	print('Simulating scan...')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, zoom_out_radii, brightnesses = brightnesses, is_label=False, num_workers=num_workers)
	plot_with_side_view(canvas, f'output/figs/sim2/steps/2.png', cmap=GFP_CMAP)
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel) # blur with gaussian filter
	plot_with_side_view(canvas, f'output/figs/sim2/steps/3.png', cmap=GFP_CMAP)
	print(this_kernel.max(), this_kernel.min())
	print(canvas.max(), canvas.min())
	canvas = convolve(canvas, this_kernel, mode='same') # apply psf
	plot_with_side_view(canvas, f'output/figs/sim2/steps/4.png', cmap=GFP_CMAP)
	print(canvas.max(), canvas.min())
	canvas = np.array((canvas/canvas.max())*255, dtype='uint8') # reconvert to 8 bit
	canvas = crop3d(canvas, zoom_out_size) # crop to selected size to remove padding
	canvas = ndimage.zoom(canvas, zoom)# zoom back in to original size for aliasing
	print(canvas.max(), canvas.min())
	plot_with_side_view(canvas, f'output/figs/sim2/steps/5.png', cmap=GFP_CMAP)
	
	canvas = [random_noise(img, mode='gaussian', var=noise_std**2)*255 for img in canvas] # Add noise
	canvas = np.array(canvas, dtype='uint8')
	plot_with_side_view(canvas, f'output/figs/sim2/steps/6.png', cmap=GFP_CMAP)
	
	# centers=[]
	# for c in zoom_out_centers:
	# 	x,y,z = c
	# 	new_c = [(x-(pad/2))*zoom ,(y-(pad/2))*zoom ,(z-(pad/2))*zoom ]
	# 	centers.append(new_c)
	# centers = np.array(centers)

	if make_label:
		# draw label heatmap from centers
		label = np.zeros(canvas_size, dtype='float64')

		print('Simulating label...')

		label = draw_spheres_sliced(label, centers, radii, is_label=True, heatmap_r=heatmap_r, num_workers=num_workers)

		# centers, diameters = crop_positions_for_label(centers, canvas_size=canvas_size, label_size=canvas_size, diameters=diameters, pad=0)

		label = np.array(label, dtype='float64')
		return canvas, label, centers, diameters
	else:
		return canvas, centers, diameters


if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)

	num_workers = 16
	heatmap_r = 'radius'
	index=0

	# read huygens psf
	psf_path = Path(dataset_path) / 'Real/PSF' / 'psf_stedXY.tif'
	psf_kernel = dc.read_tif(str(psf_path))
	psf_kernel = dc.crop3d(psf_kernel, (54,16,16), (27,139,140))
	psf_kernel = psf_kernel/psf_kernel.max()

	params = dict(
			r=5,
			particle_size=0.2,
			snr=30,
			cnr=3,
			volfrac=0.25,
			brightness=200,
		)
	params['f_sigma'] = 10
	params['b_sigma'] = 15
	volfrac = params['volfrac']

	path = f'{dataset_path}/Positions/old/phi{volfrac*1000:.0f}.gsd'
	print(f'Reading: {path} at {index+300} ...')

	hoomd_positions, diameters = read_gsd(path, index+1)

	canvas, final_centers, final_diameters = simulate(canvas_size, hoomd_positions, params['r'], 
														params['particle_size'], params['brightness'], params['cnr'],
														params['snr'], diameters=diameters, make_label=False, heatmap_r=heatmap_r, 
														num_workers=num_workers, psf_kernel=psf_kernel)

	plot_with_side_view(canvas, f'output/figs/sim2/steps/7.png', cmap=GFP_CMAP)
	