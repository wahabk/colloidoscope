"""
Colloidoscope simulator

This creates simulations of small particles with a bad point spread function blur, to see how to use it please see `scripts/simulate/sim.py`

The usual method of creating simulations is a class-based approach, however, 
I chose a functional method because I found it more intuitive to use NJIT compiling and to mulithread

"""

from .hoomd_sim_positions import convert_hoomd_positions
from concurrent.futures import ProcessPoolExecutor
# from mpi4py.futures import MPIPoolExecutor
from skimage.util import random_noise
from scipy import ndimage
import numpy as np
import random
import random
import math
from scipy.signal import convolve
from tqdm import tqdm
# import psf
from perlin_numpy import generate_perlin_noise_3d

import copy

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
						new_slice[i,j] = 255 * np.exp(-(((dist*math.sqrt(3)))**2.)/((2. * heatmap_r)**2.))

	return new_slice

def draw_spheres_sliced(canvas, centers, radii, brightnesses=None, is_label=False, heatmap_r='radius', num_workers=2):
	from numba import njit
	new_canvas = []

	if is_label == False:
		args = [(s, z, radii, centers, brightnesses) for z, s in enumerate(canvas)]

		print(canvas.shape, centers.shape, np.shape(radii))
		jitted_draw_slice = njit(draw_slice)
		with tqdm(total=len(args)) as pbar:
			with ProcessPoolExecutor(max_workers=num_workers) as pool:
				for i in pool.map(jitted_draw_slice, args):
					new_canvas.append(i)
					pbar.update(1)

	elif is_label == True:
		is_seg = False
		if heatmap_r == 'radius'			: heatmap_radii = radii 
		elif isinstance(heatmap_r, int)		: heatmap_radii = [heatmap_r for i in radii]
		elif 'seg' in heatmap_r 			: heatmap_radii = [heatmap_r[-1] for i in radii]; is_seg = True
		else: raise Exception('invalid heatmap type')
		args = [(s, z, heatmap_radii, centers, is_seg) for z, s in enumerate(canvas)]

		print(canvas.shape, centers.shape, np.shape(radii))
		jitted_draw_label_slice = njit(draw_label_slice)
		with tqdm(total=len(args)) as pbar:
			with ProcessPoolExecutor(max_workers=num_workers) as pool:
				for i in pool.map(jitted_draw_label_slice, args):
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

def crop_positions_for_label(centers, canvas_size, label_size, diameters, pad=0):

	zdiff = (canvas_size[0] - label_size[0])/2
	xdiff = (canvas_size[1] - label_size[1])/2
	ydiff = (canvas_size[2] - label_size[2])/2
	centers = centers - [zdiff, xdiff, ydiff]

	
	indices = []
	# pad = 0
	for idx, c in enumerate(centers):
		if pad<=c[0]<=(label_size[0]-pad) and pad<=c[1]<=(label_size[1]-pad) and pad<=c[2]<=(label_size[2]-pad):
			indices.append(idx)

	final_centers = centers[indices]
	final_diameters = diameters[indices]

	return final_centers, final_diameters


def make_background(canvas_shape, octaves, brightness_mean, brightness_std, tileable=(False, False, False), dtype='float32'):
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
			snr:float, f_sigma:float=10, b_sigma:float=5, diameters=np.ndarray([]), make_label:bool=True, 
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
	canvas = make_background(zoom_out_size_padded, 2, 1, b_sigma/255, dtype='uint8')*b_mean
	
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
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel) # blur with gaussian filter
	canvas = convolve(canvas, this_kernel, mode='same') # apply psf
	canvas = np.array((canvas/canvas.max())*255, dtype='uint8') # reconvert to 8 bit
	canvas = crop3d(canvas, zoom_out_size) # crop to selected size to remove padding
	canvas = ndimage.zoom(canvas, zoom)# zoom back in to original size for aliasing
	
	canvas = [random_noise(img, mode='gaussian', var=noise_std**2)*255 for img in canvas] # Add noise
	canvas = np.array(canvas, dtype='uint8')
	
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

