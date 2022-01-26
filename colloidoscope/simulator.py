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
def gaussian(x, mu, sig, peak=1.):
	# peak = 1./(np.sqrt(2.*np.pi*sig)) # regularisation term to make area under gauss = 1
	return peak * np.exp(-((x-mu)**2.)/((2. * sig)**2.))

@njit()
def draw_slice(args):
	# extract args
	s, z, radii, centers, brightnesses, is_label = args
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
					if is_label:
						new_slice[i,j] = gaussian(dist*2, 0, r, peak=255)
					else:
						new_slice[i,j] = brightness
	return new_slice

def draw_spheres_sliced(canvas, centers, radii, brightnesses=None, is_label=False, num_workers=2):
	new_canvas = []

	if brightnesses == None or is_label == True:
		brightnesses = [255 for _ in centers]

	args = [(s, z, radii, centers, brightnesses, is_label) for z, s in enumerate(canvas)]

	print(canvas.shape, centers.shape, np.shape(radii))

	with ProcessPoolExecutor(max_workers=num_workers) as pool:
		for i in pool.map(draw_slice, args):
			new_canvas.append(i)

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

def simulate(canvas_size:list, centers:np.ndarray, r:int,
			xy_gauss:int, z_gauss:int, min_brightness:int, max_brightness:int, noise:float, make_label=True, diameters=np.ndarray([]), num_workers=2):
	brightnesses = [random.randrange(min_brightness, max_brightness) for _ in centers]
	zoom = 0.5
	gauss_kernel = (z_gauss, xy_gauss, xy_gauss)
	# make bigger padded canvas
	# this is to allow the gaussian blur to work on the edges
	
	
	# zoom out to large image and positions
	# later we zoom back in to add aliasing
	# TODO make bigger canvas then crop
	zoom_out_r = int(r/zoom)
	zoom_out_size = [int(c/zoom) for c in canvas_size]
	bigger_canvas_size = [int(c*1) for c in zoom_out_size]
	canvas = np.zeros(bigger_canvas_size, dtype='uint8')
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom,y/zoom,z/zoom]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)

	radii = [(d/2) for d in diameters]
	zoom_out_radii = [r/zoom for r in radii]

	# draw spheres slice by slice
	print('Simulating scan...')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, zoom_out_radii, brightnesses = brightnesses, is_label=False, num_workers=num_workers)
	# zoom back in for aliasing
	canvas = crop3d(canvas, zoom_out_size)
	canvas = ndimage.zoom(canvas, zoom)
	# blur with gaussian filter
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel)
	# Add noise
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] 
	
	# crop to selected size
	canvas = np.array(canvas, dtype='uint8')
	
	if make_label:
		# draw label heatmap from centers
		label = np.zeros(canvas_size, dtype='float64')
		print('Simulating label...')
		label = draw_spheres_sliced(label, centers, radii, is_label=True, num_workers=num_workers)
		# print(label.shape, label.max(), label.min(), r, centers.shape, num_workers, )
		label = np.array(label ,dtype='float64')
		return canvas, label
	else:
		return canvas


def simulate_img3d(canvas_size, zoom, gauss, noise = 0.09, volfrac=0.3):
	'''
	DEPRECATED

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
	hoomd_positions = hooomd_sim_positions(phi=volfrac, canvas_size=canvas_size)
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
	label = draw_spheres_sliced(label, centers, r, brightness=255, is_label=True)

	# zoom back in for aliasing
	canvas = ndimage.zoom(canvas, zoom)
	
	if gauss: canvas = ndimage.gaussian_filter(canvas, gauss)
	# label = ndimage.gaussian_filter(canvas, [3,3,3])
	# Add noise
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] 
	canvas = np.array(canvas,dtype='uint8')
	label = np.array(label,dtype='uint8')/label.max()

	return canvas, centers, label