from colloidoscope import DeepColloid
from colloidoscope.simulator import draw_spheres_sliced, crop_positions_for_label, make_background, crop3d
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
import random
import numpy as np
from skimage.util import random_noise
from scipy import ndimage
from scipy.signal import convolve
import psf
import matplotlib.pyplot as plt

def simulate(canvas_size:list, centers:np.ndarray, r:int,
			blur_kernel:np.ndarray, min_brightness:int, max_brightness:int, 
			noise:float, make_label:bool=True, label_size:list=(64,64,64), diameters=np.ndarray([]), num_workers=2):
	'''
	This will only work for a perfect cube eg 64x64x64
	'''


	brightnesses = [random.randrange(min_brightness, max_brightness) for _ in centers]
	zoom = 0.5
	pad = 32
	gauss_kernel = (2, 2, 2)
	# make bigger padded canvas
	# this is to allow the blur to work on the edges
	
	
	# zoom out to large image and positions
	# later we zoom back in to add aliasing
	zoom_out_size = [int(c/zoom) for c in canvas_size]
	zoom_out_size_padded = [c+pad for c in zoom_out_size]
	# canvas = np.zeros(zoom_out_size_padded, dtype='uint8')
	canvas = make_background(zoom_out_size_padded, 4, random.randint(0,50), dtype='uint8') 

	plot_with_side_view(canvas, 'output/figs/simulation/1.png')

	# canvas = np.zeros(zoom_out_size_padded, dtype='uint8')
	print('background', canvas.max())
	
	# convert centers to zoom out
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom + pad/2 ,y/zoom + pad/2 ,z/zoom + pad/2 ]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)

	radii = [(d*r) for d in diameters]
	zoom_out_radii = [(r/zoom) for r in radii]


	# draw spheres slice by slice
	print('Simulating scan...')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, zoom_out_radii, brightnesses = brightnesses, is_label=False, num_workers=num_workers)
	plot_with_side_view(canvas, 'output/figs/simulation/2.png')
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel) # blur with gaussian filter
	plot_with_side_view(canvas, 'output/figs/simulation/3.png')
	canvas = convolve(canvas, blur_kernel, mode='same') # apply psf
	canvas = np.array((canvas/canvas.max())*255, dtype='uint8') # reconvert to 8 bit
	canvas = crop3d(canvas, zoom_out_size) # crop to selected size to remove padding
	canvas = ndimage.zoom(canvas, zoom)# zoom back in to original size for aliasing
	plot_with_side_view(canvas, 'output/figs/simulation/4.png')
	
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] # Add noise
	plot_with_side_view(canvas, 'output/figs/simulation/5.png')
	
	canvas = np.array(canvas, dtype='uint8')
	
	if make_label:
		# draw label heatmap from centers
		label = np.zeros(label_size, dtype='float64')

		diff = (canvas.shape[0] - label.shape[0])/2
		final_centers = centers - diff

		print('Simulating label...')
		radii = [(d*r) for d in diameters]
		label = draw_spheres_sliced(label, final_centers, radii, is_label=True, num_workers=num_workers)
		plot_with_side_view(label, 'output/figs/simulation/label.png')
		final_centers, final_diameters = crop_positions_for_label(final_centers, label_size, diameters, r*2)

		# print(label.shape, label.max(), label.min(), r, centers.shape, num_workers, )
		label = np.array(label ,dtype='float64')
		return canvas, label, final_centers, final_diameters
	else:
		return canvas

def plot_with_side_view(scan, path):
	projection = np.max(scan, axis=0)
	side_projection = np.max(scan, axis=1)
	# side_projection = np.rot90(side_projection)
	sidebyside = np.concatenate((projection, side_projection), axis=0)
	plt.imsave(path, sidebyside, cmap='gray')
	plt.clf()

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(64,64,64)
	num_workers=10
	r = 5
	min_brightness = 80
	max_brightness = 200
	psf_zoom = 0.9
	noise = 0.02
	volfrac = 0.2

	args = dict(shape=(64, 64), dims=(4, 4), ex_wavelen=488, em_wavelen=520,
				num_aperture=1.2, refr_index=1.4,
				pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
	obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
	kernel = obsvol.volume()

	kernel = ndimage.zoom(kernel, psf_zoom)

	

	path = f'{dataset_path}/Positions/poly/phi_{volfrac*1000:.0f}_poly.gsd'
	hoomd_positions, diameters = read_gsd(path, 1)
	
	print(diameters.shape)
	centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)

	canvas, label, final_centers, final_diameters = simulate(canvas_size, 
																centers, 
																r, 
																kernel, 
																min_brightness, 
																max_brightness,
																noise, 
																make_label=True, 
																label_size=label_size, 
																diameters=diameters, 
																num_workers=num_workers)
	

	# plot_with_side_view(canvas, f'output/figs/simulation/{i}.png')
	