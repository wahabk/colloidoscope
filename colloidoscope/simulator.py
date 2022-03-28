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
from scipy.signal import convolve
from tqdm import tqdm

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
				
				if is_label:
					r=5
					if dist <= r:
						new_slice[i,j] = gaussian(dist*3, 0, r, peak=255)
				else:
					if dist <= r:
						new_slice[i,j] = brightness
	return new_slice

def draw_spheres_sliced(canvas, centers, radii, brightnesses=None, is_label=False, num_workers=2):
	new_canvas = []

	if brightnesses == None or is_label == True:
		brightnesses = [255 for _ in centers]

	args = [(s, z, radii, centers, brightnesses, is_label) for z, s in enumerate(canvas)]

	print(canvas.shape, centers.shape, np.shape(radii))
	with tqdm(total=len(args)) as pbar:
		with ProcessPoolExecutor(max_workers=num_workers) as pool:
			for i in pool.map(draw_slice, args):
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

def crop_positions_for_label(centers, canvas_size, diameters, diameter=10):
	indices = []
	# divisor = -4
	# diff = diameter / divisor
	diff = 0
	for idx, c in enumerate(centers):
		if diff<c[0]<(canvas_size[0]-diff) and diff<c[1]<(canvas_size[1]-diff) and diff<c[2]<(canvas_size[2]-diff):
			indices.append(idx)

	centers = centers[indices]
	diameters = diameters[indices]

	return centers, diameters


def make_background(canvas_size, octaves, brightness, dtype='uint8'):
	
	from perlin_noise import PerlinNoise

	noise = PerlinNoise(octaves=octaves)
	zpix, xpix, ypix = canvas_size
	pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)] 
	pic = [pic for _ in range(zpix)]

	pic = np.array(pic)
	pic = pic-pic.min()
	pic = pic/pic.max()
	pic = pic*brightness

	pic = np.array(pic, dtype=dtype)
	return pic


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
	# canvas = make_background(zoom_out_size_padded, 4, random.randint(0,50), dtype='uint8') 
	canvas = np.zeros(zoom_out_size_padded, dtype='uint8')
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
	canvas = ndimage.gaussian_filter(canvas, gauss_kernel) # blur with gaussian filter
	canvas = convolve(canvas, blur_kernel, mode='same') # apply psf
	canvas = np.array((canvas/canvas.max())*255, dtype='uint8') # reconvert to 8 bit
	canvas = crop3d(canvas, zoom_out_size) # crop to selected size to remove padding
	canvas = ndimage.zoom(canvas, zoom)# zoom back in to original size for aliasing
	
	canvas = [random_noise(img, mode='gaussian', var=noise)*255 for img in canvas] # Add noise
	
	canvas = np.array(canvas, dtype='uint8')
	
	if make_label:
		# draw label heatmap from centers
		label = np.zeros(label_size, dtype='float64')

		diff = (canvas.shape[0] - label.shape[0])/2
		final_centers = centers - diff

		print('Simulating label...')
		radii = [(d*r) for d in diameters]
		label = draw_spheres_sliced(label, final_centers, radii, is_label=True, num_workers=num_workers)
		final_centers, final_diameters = crop_positions_for_label(final_centers, label_size, diameters, r*2)

		# print(label.shape, label.max(), label.min(), r, centers.shape, num_workers, )
		label = np.array(label ,dtype='float64')
		return canvas, label, final_centers, final_diameters
	else:
		return canvas

