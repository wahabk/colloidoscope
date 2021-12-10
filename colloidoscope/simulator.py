from .hoomd_sim_positions import hooomd_sim_positions, convert_hoomd_positions
from concurrent.futures import ProcessPoolExecutor
# from mpi4py.futures import MPIPoolExecutor
from skimage.util import random_noise
from numba import njit
from scipy import ndimage
import numpy as np
import random
import random
import math

@njit()
def gaussian(x, mu, sig):
	# gaussian equation for position heatmap generation
	# numba decorator required for multithreading
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

@njit()
def draw_slice(args):
	# extract args
	s, z, r, centers, brightness, is_label = args
	#initiate new slice to be drawn
	new_slice = s
	#for each sphere check if this pixel is inside it
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				cz, cy, cx = center
				# euclidean distance
				dist = math.sqrt((z - cz)**2 + (i - cy)**2 + (j - cx)**2)
				if dist <= r*2:
					if is_label:
						new_slice[i,j] = gaussian(dist, 0, r) * brightness
					else:
						new_slice[i,j] = brightness
	return new_slice

def draw_spheres_sliced(canvas, centers, r, brightness=255, is_label=False, num_workers=2):
	new_canvas = []

	args = [(s, z, r, centers, brightness, is_label) for z, s in enumerate(canvas)]

	with ProcessPoolExecutor(max_workers=num_workers) as pool:
		for i in pool.map(draw_slice, args):
			new_canvas.append(i)

	canvas = list(new_canvas)

	if is_label: 
		canvas = np.array(canvas, dtype='float32')
		canvas = canvas/canvas.max()
	else: canvas = np.array(canvas, dtype='uint8')
	return canvas

def shake(centers, magnitude):
	new_centers = []
	for cz, cy, cx in centers:
		cz = cz + random.uniform(-magnitude, magnitude)
		cy = cy + random.uniform(-magnitude, magnitude)
		cx = cx + random.uniform(-magnitude, magnitude)
		new_centers.append([cz, cy, cx])
	return np.array(new_centers)

def simulate(canvas_size:list, centers:np.ndarray, r:int,
			xy_gauss:int, z_gauss:int, brightness:int, noise:float, make_label=True, num_workers=2):
	zoom = 0.5
	gauss_kernel = (z_gauss, xy_gauss, xy_gauss)
	
	# zoom out to large image and positions
	# later we zoom back in to add aliasing
	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out, dtype='uint8')
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom,y/zoom,z/zoom]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)


	label = np.zeros(canvas_size, dtype='uint8')

	# draw spheres slice by slice
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, r, brightness = brightness, is_label=False, num_workers=num_workers)
	# draw label heatmap from centers
	label = draw_spheres_sliced(label, centers, r, is_label=True, num_workers=num_workers)
	print(label.shape, label.max(), label.min(), r, centers.shape, num_workers, )

	# zoom back in for aliasing
	canvas = ndimage.zoom(canvas, zoom)
	# blur with gaussian filter
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel)
	# Add noise
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] 
	
	canvas = np.array(canvas,dtype='uint8')
	label = np.array(label ,dtype='float32')
	
	if make_label:
		return canvas, label
	else:
		return canvas


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
	r = random.randrange(2,12)
	brightness = random.randrange(150,250)
	# centers = pycrusher.gen_rand_centers(volfrac=volfrac, canvas_size=canvas_size,  diameter=r*2)
	hoomd_positions = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size, diameter=r*2)
	centers = convert_hoomd_positions(hoomd_positions, canvas_size=canvas_size, diameter=r*2)
	
	# zoom out to large image and positions
	# later we zoom back in to add aliasing
	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out, dtype='uint8')
	label = np.zeros(canvas_size, dtype='uint8')
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom,y/zoom,z/zoom]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)
	
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, r, brightness = brightness)
	label = draw_spheres_sliced(label, centers, r, is_label=True)

	# zoom back in for aliasing
	canvas = ndimage.zoom(canvas, zoom)
	
	if gauss: canvas = ndimage.gaussian_filter(canvas, gauss)
	# label = ndimage.gaussian_filter(canvas, [3,3,3])
	# Add noise
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] 
	canvas = np.array(canvas,dtype='uint8')
	label = np.array(label,dtype='uint8')/label.max()

	return canvas, centers, label