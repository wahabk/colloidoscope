from numba import njit
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random
import os
#255x255x128
from concurrent.futures import ProcessPoolExecutor
from skimage.util import random_noise
import time
import random
import os
from crusher import pycrusher
from scipy import ndimage

print(os.cpu_count())

@njit()
def draw_slice(args):
	s, z, r, centers, brightness, is_label = args
	new_slice = s
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				cz, cy, cx = center
				dist = (z - cz)**2 + (i - cy)**2 + (j - cx)**2
				if dist <= r**2:
					if is_label:
						new_slice[i,j] = (1-(dist/r**2))**2 * 255
					else:
						new_slice[i,j] = brightness
	return new_slice

def draw_spheres_sliced(canvas, centers, r, brightness=255, is_label=False):
	new_canvas = []

	args = [(s, z, r, centers, brightness, is_label) for z, s in enumerate(canvas)]

	with ProcessPoolExecutor(max_workers=12) as pool:
		for i in pool.map(draw_slice, args):
			new_canvas.append(i)

	canvas = list(new_canvas)
	canvas = np.array(canvas, dtype='uint8')
	return canvas

def shake(centers, magnitude):
	new_centers = []
	for cz, cy, cx in centers:
		cz = cz + random.uniform(-magnitude, magnitude)
		cy = cy + random.uniform(-magnitude, magnitude)
		cx = cx + random.uniform(-magnitude, magnitude)
		new_centers.append([cz, cy, cx])
	return np.array(new_centers)

def simulate_img3d(canvas_size, zoom, gauss, noise = 0.09, volfrac=0.3):
	'''
	Simulate 3d image of colloids

	centers : list of centers of each particle x,y
	r : radius of particles
	zoom : factor to zoom image out before generation then zoom in as a way to add aliasing
	gauss : kernel for gaussian noise

	TODO add background
	'''
	
	''' make half polydisperse '''
	r = random.randrange(5,6)
	brightness = random.randrange(200,250)
	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out, dtype='uint8')
	label = np.zeros(canvas_size, dtype='uint8')
	centers = pycrusher.gen_rand_centers(volfrac=volfrac, canvas_size=canvas_size,  diameter=r*2)
	
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom,y/zoom,z/zoom]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)

	print('Number of particles: ', len(centers))

	print('making image')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, r, brightness = brightness)
	print('making label')
	label = draw_spheres_sliced(label, centers, r, is_label=True)

	canvas = ndimage.zoom(canvas, zoom)
	
	if gauss: canvas = ndimage.gaussian_filter(canvas, gauss)
	# label = ndimage.gaussian_filter(canvas, [3,3,3])
	# Add noise
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] 
	canvas = np.array(canvas,dtype='uint8')
	label = np.array(label,dtype='uint8')

	return canvas, centers, label

