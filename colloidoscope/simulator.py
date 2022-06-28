"""
Colloidoscope simulator

This creates simulations of small particles with a bad point spread function blur, to see how to use it please see `scripts/sim.py`

The usual method of creating simulations is a class-based approach, however, 
I chose a functional method because I found it more intuitive to use NJIT compiling and to mulithread

"""

from .hoomd_sim_positions import convert_hoomd_positions
from concurrent.futures import ProcessPoolExecutor
# from mpi4py.futures import MPIPoolExecutor
from skimage.util import random_noise
from numba import njit
from scipy import ndimage
import numpy as np
import random
import random
import math
from scipy.signal import convolve
from tqdm import tqdm
import psf
from perlin_numpy import generate_perlin_noise_3d
from pathlib2 import Path

@njit()
def gaussian(x, mu, sig, peak=1.):
	# peak = 1./(np.sqrt(2.*np.pi*sig)) # regularisation term to make area under gauss = 1
	return peak * np.exp(-((x-mu)**2.)/((2. * sig)**2.))

@njit()
def draw_slice(args):
	# extract args
	s, z, radii, centers, brightnesses = args
	#initiate new slice to be drawn
	new_slice = s
	#for each sphere check if this pixel is inside it
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				cz, cy, cx = center
				r = radii[k]
				brightness = brightnesses[k]
				# euclidean distance
				dist = math.sqrt((z - cz)**2 + (i - cy)**2 + (j - cx)**2)
				
				if dist <= r:
					new_slice[i,j] = brightness
	return new_slice

@njit()
def draw_label_slice(args):
	# extract args
	s, z, heatmap_radii, centers, is_seg = args
	#initiate new slice to be drawn
	new_slice = s
	#for each sphere check if this pixel is inside it
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				cz, cy, cx = center
				heatmap_r = heatmap_radii[k]
				# euclidean distance
				dist = math.sqrt((z - cz)**2 + (i - cy)**2 + (j - cx)**2)
				
				if dist <= heatmap_r:
					if is_seg:
						new_slice[i,j] = 255
					else:
						new_slice[i,j] = gaussian(dist*3, 0, heatmap_r, peak=255)

	return new_slice

def draw_spheres_sliced(canvas, centers, radii, brightnesses=None, is_label=False, heatmap_r='radius', num_workers=2):
	new_canvas = []

	if is_label == False:
		args = [(s, z, radii, centers, brightnesses) for z, s in enumerate(canvas)]

		print(canvas.shape, centers.shape, np.shape(radii))
		with tqdm(total=len(args)) as pbar:
			with ProcessPoolExecutor(max_workers=num_workers) as pool:
				for i in pool.map(draw_slice, args):
					new_canvas.append(i)
					pbar.update(1)

	elif is_label == True:
		is_seg = False
		if heatmap_r == 'radius'			: heatmap_radii = radii 
		elif isinstance(heatmap_r, int)		: heatmap_radii = [heatmap_r for i in radii]
		elif 'seg' in heatmap_r 			: heatmap_radii = [r-int(heatmap_r[-1]) for r in radii]; is_seg = True
		else: raise Exception('invalid heatmap type')
		args = [(s, z, heatmap_radii, centers, is_seg) for z, s in enumerate(canvas)]

		print(canvas.shape, centers.shape, np.shape(radii))
		with tqdm(total=len(args)) as pbar:
			with ProcessPoolExecutor(max_workers=num_workers) as pool:
				for i in pool.map(draw_label_slice, args):
					new_canvas.append(i)
					pbar.update(1)

	canvas = list(new_canvas)

	if is_label: 
		canvas = np.array(canvas, dtype='float64')
		canvas = canvas/canvas.max()
	else: 
		canvas = np.array(canvas, dtype='uint8')
	return canvas

def crop3d(array, roiSize, center=None):
	roiZ, roiY, roiX = roiSize
	zl = int(roiZ / 2)
	yl = int(roiY / 2)
	xl = int(roiX / 2)

	if center == None:
		center = [int(c/2) for c in array.shape]

	z, y, x = center
	z, y, x = int(z), int(y), int(x)

	new_array = array[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl]
	return new_array

def shake(centers, magnitude):
	new_centers = []
	for cz, cy, cx in centers:
		cz = cz + random.uniform(-magnitude, magnitude)
		cy = cy + random.uniform(-magnitude, magnitude)
		cx = cx + random.uniform(-magnitude, magnitude)
		new_centers.append([cz, cy, cx])
	return np.array(new_centers)

def crop_positions_for_label(centers, canvas_size, label_size, diameters):
	indices = []

	pad = 0
	for idx, c in enumerate(centers):
		if pad<=c[0]<=(label_size[0]-pad) and pad<=c[1]<=(label_size[1]-pad) and pad<=c[2]<=(label_size[2]-pad):
			indices.append(idx)

	centers = centers[indices]
	diameters = diameters[indices]

	zdiff = (canvas_size[0] - label_size[0])/2
	xdiff = (canvas_size[1] - label_size[1])/2
	ydiff = (canvas_size[2] - label_size[2])/2

	centers = centers - [zdiff, xdiff, ydiff]

	return centers, diameters


def make_background(canvas_shape, octaves, brightness_mean, brightness_std, tileable=(False, False, False), dtype='uint8'):
	array = generate_perlin_noise_3d(canvas_shape, (octaves, octaves, octaves), tileable=tileable)

	# normalise to 0 - 1
	array = array-array.min()
	array = array/array.max()

	# find linear transformation to shift mean and std to required
	a = math.sqrt( brightness_std ** 2 / array.std() ** 2 )
	b = brightness_mean - (a * array.mean())
	array = a * array + b

	array = ndimage.gaussian_filter(array, (3,3,3))

	array = np.array(array, dtype=dtype)
	
	return array


def simulate(canvas_size:list, hoomd_positions:np.ndarray, r:int,
			particle_size:float, f_mean:float, cnr:float,
			snr:float, f_sigma:float=30, b_sigma:float=20, diameters=np.ndarray([]), make_label:bool=True, 
			label_size:list=(64,64,64), heatmap_r='radius', num_workers=2, psf_kernel = 'standard'):
	'''
	particle size in um

	heatmap_r can be 'radius' to equal the particle or can be a constant int

	This will only work for a perfect cube eg 64x64x64 not cuboids

	psf_kernel = 'standard'

	TODO add sim seg instead of heatmap
	'''
	
	centers, diameters = convert_hoomd_positions(hoomd_positions, canvas_size, diameter=r*2, diameters=diameters)


	b_mean = abs(cnr * b_sigma + f_sigma)

	brightnesses = [random.gauss(f_mean, f_sigma) for _ in centers]
	zoom = 0.5
	pad = 80
	gauss_kernel = (2, 2, 2)
	noise = (f_mean / (snr * 255))/10

	# zoom out to large image and positions
	# later we zoom back in to add aliasing
	zoom_out_size = [int(c/zoom) for c in canvas_size]
	# make bigger padded canvas
	# this is to allow the blur to work on the edges
	zoom_out_size_padded = [c+pad for c in zoom_out_size]
	canvas = make_background(zoom_out_size_padded, 4, b_mean, b_sigma, dtype='uint8') 
	canvas = np.zeros(zoom_out_size_padded, dtype='uint8')
	
	# convert centers to zoom out size
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom + pad/2 ,y/zoom + pad/2 ,z/zoom + pad/2 ]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)

	radii = [(d*r) for d in diameters]
	zoom_out_radii = [(r/zoom) for r in radii]

	# create PSF
	if psf_kernel is 'standard':
		args = dict(shape=(64, 64), dims=(particle_size, particle_size), ex_wavelen=488, em_wavelen=520, num_aperture=1.2, refr_index=1.4, pinhole_radius=0.9, pinhole_shape='round', magnification = 100)
		obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
		psf_kernel = obsvol.volume()
		psf_kernel = ndimage.zoom(psf_kernel, 0.375)
		

	else:
		psf_kernel = psf_kernel 
		psf_zoom = -2*particle_size + 2.4
		if psf_zoom < 0.1: psf_zoom=0.1

		psf_kernel = ndimage.zoom(psf_kernel, psf_zoom)



	# draw spheres slice by slice
	print('Simulating scan...')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, zoom_out_radii, brightnesses = brightnesses, is_label=False, num_workers=num_workers)
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel) # blur with gaussian filter
	canvas = convolve(canvas, psf_kernel, mode='same') # apply psf
	canvas = np.array((canvas/canvas.max())*255, dtype='uint8') # reconvert to 8 bit
	canvas = crop3d(canvas, zoom_out_size) # crop to selected size to remove padding
	canvas = ndimage.zoom(canvas, zoom)# zoom back in to original size for aliasing
	
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] # Add noise
	
	canvas = np.array(canvas, dtype='uint8')
	
	if make_label:
		# draw label heatmap from centers
		label = np.zeros(label_size, dtype='float64')

		# TODO this is the limit for label having to be square not rect
		diff = (canvas.shape[0] - label.shape[0])/2
		final_centers = centers - diff # if the label is smaller than the image then shift the difference

		print('Simulating label...')
		radii = [(d*r) for d in diameters]

		label = draw_spheres_sliced(label, final_centers, radii, is_label=True, heatmap_r=heatmap_r, num_workers=num_workers)
		final_centers, final_diameters = crop_positions_for_label(final_centers, canvas_size, label_size, diameters)

		# print(label.shape, label.max(), label.min(), r, centers.shape, num_workers, )
		label = np.array(label ,dtype='float64')
		return canvas, label, final_centers, final_diameters
	else:
		return canvas

