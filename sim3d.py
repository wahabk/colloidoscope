from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, spatial
from mainviewer import mainViewer
from concurrent.futures import ProcessPoolExecutor
import random
from math import sqrt
from poisson_disc_sampling import poisson_disc_sampling
import os
#255x255x128
from concurrent.futures import ProcessPoolExecutor
from skimage.util import random_noise
import time
import random
import os

print(os.cpu_count())

@njit()
def draw_slice(args):
	s, z, radii, centers, brightness = args
	new_slice = s
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for k, center in enumerate(centers):
				r = radii[k]
				cz, cy, cx = center
				dist = (z - cz)**2 + (i - cy)**2 + (j - cx)**2
				if dist <= r**2:
					new_slice[i,j] = 255
	return new_slice

def draw_spheres_sliced(canvas, centers, radii, is_label=False):
	new_canvas = []

	if is_label: brightness = 255
	else: brightness = random.randrange(100,255)

	args = [(s, z, radii, centers, brightness) for z, s in enumerate(canvas)]

	with ProcessPoolExecutor(max_workers=12) as pool:
		for i in pool.map(draw_slice, args):
			new_canvas.append(i)

	canvas = list(new_canvas)
	canvas = np.array(canvas, dtype='uint8')
	return canvas

def simulate_img3d(canvas_size, zoom, gauss, noise = 0.09, k=50):
	'''
	Simulate 3d image of colloids

	centers : list of centers of each particle x,y
	r : radius of particles
	zoom : factor to zoom image out before generation then zoom in as a way to add aliasing
	gauss : kernel for gaussian noise

	TODO add background
	'''
	
	''' make half polydisperse '''
	polydisperse = bool(random.getrandbits(1)) #random bool
	if polydisperse:
		min_dist = 13
	else:
		r = random.randrange(7,13)
		min_dist=r
	

	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out, dtype='uint8')
	label = np.zeros(canvas_size, dtype='uint8')
	centers = poisson_disc_sampling(min_dist, np.array(canvas_size), k)
	
	if polydisperse:
		# half the time polydisperse, make particles of diff sizes
		radii = [random.randrange(7,13) for i in range(len(centers))]
	else:
		# half the time keep all colloid 
		radii = [r for i in range(len(centers))]
	
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom,y/zoom,z/zoom]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)

	print('Number of particles: ', len(centers))

	print('making image')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, radii)
	print('making label')
	label = draw_spheres_sliced(label, centers, radii, is_label=True)

	canvas = ndimage.zoom(canvas, zoom)
	
	if gauss: canvas = ndimage.gaussian_filter(canvas, gauss)
	
	# Add noise
	canvas = [random_noise(img, mode='gaussian',var=noise)*255 for img in canvas]
	canvas = np.array(canvas,dtype='uint8')

	return canvas, centers, label

def vol_frac(centers, r, canvas_size):
	vol = (4/3)*  np.pi * r**3
	num = len(centers)
	z, x, y = canvas_size
	volsys = z*x*y
	volfrac = (vol * num) / volsys
	return volfrac

def get_gr(positions, cutoff, bins, minimum_gas_number=1e4):
	# from yushi yang
	bins = np.linspace(0, cutoff, bins)
	drs = bins[1:] - bins[:-1]
	distances = pdist(positions).ravel()

	if positions.shape[0] < minimum_gas_number:
		rg_hists = []None, None, None, k=k, noise=noise)
		for i in range(int(minimum_gas_number) // positions.shape[0] + 2):
			random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
			rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]
			rg_hists.append(rg_hist)
		rg_hist = np.mean(rg_hists, 0)

	else:
		random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
		rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]

	hist = np.histogram(distances, bins=bins)[0]
	hist = hist / rg_hist # pdfs
	hist[np.isnan(hist)] = 0
	bin_centres = (bins[1:] + bins[:-1]) / 2
	return bin_centres, hist # as x, y






















