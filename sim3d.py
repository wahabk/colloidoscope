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

@njit
def draw_slice(args):
	s, z, centers, r = args
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for center in centers:
				cz, cy, cx = center
				dist = (z - cz)**2 + (i - cy)**2 + (j - cx)**2
				if dist <= r**2:
					s[i,j] = 255
	return s

def draw_spheres_sliced(canvas, centers, r):
	new_canvas = []

	args = [(s, z, centers, r) for z, s in enumerate(canvas)]

	with ProcessPoolExecutor(max_workers=10) as pool:
		new_canvas = pool.map(draw_slice, args)

	canvas = list(new_canvas)
	canvas = np.array(canvas, dtype='uint8')
	return canvas

def simulate_img3d(canvas_size, r, min_dist, zoom, gauss, noise = 0.09, k=50):
	'''
	Simulate 3d image of colloids

	centers : list of centers of each particle x,y
	r : radius of particles
	zoom : factor to zoom image out before generation then zoom in as a way to add aliasing
	gauss : kernel for gaussian noise

	TODO add background
	'''

	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out, dtype='uint8')
	label = np.zeros(canvas_size, dtype='uint8')
	centers = poisson_disc_sampling(min_dist, np.array(canvas_size), k)
	
	zoom_out_centers=[]
	for c in centers:
		x,y,z = c
		new_c = [x/zoom,y/zoom,z/zoom]
		zoom_out_centers.append(new_c)
	zoom_out_centers = np.array(zoom_out_centers)
	v = vol_frac(centers, r, canvas_size)
	print('estimated volfrac: ', v)
	print('Number of particles: ', len(centers))

	print('making image')
	canvas = draw_spheres_sliced(canvas, zoom_out_centers, r)
	print('making label')
	label = draw_spheres_sliced(label, centers, r)

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
		rg_hists = []
		for i in range(int(minimum_gas_number) // positions.shape[0] + 2):
			random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
			rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]
			rg_hists.append(rg_hist)
		rg_hist = np.mean(rg_hists, 0)

	else:
		random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
		rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]

	hist = np.histogram(distances, bins=bins)[0]
	hist = hist / rg_hist
	hist[np.isnan(hist)] = 0
	centres = (bins[1:] + bins[:-1]) / 2
	return centres, hist



if __name__ == "__main__":
	canvas_size=(32,256,256)
	r = 10
	zoom = 0.8
	gauss = (7,3,3)
	min_dist = 2*r
	k = 30


	canvas, positions, label = simulate_img3d(canvas_size, r, min_dist, zoom, gauss, k=30)
	mainViewer(canvas, positions=positions)
	mainViewer(label, positions=positions)

	print(canvas.shape, np.mean(canvas), np.std(canvas), np.min(canvas), np.max(canvas))

	# metadata = [canvas_size, r, zoom, gauss, min_dist, k]
	# metadatadict = {}
	# [metadatadict{str(var.__name__ ): var} for var in metadata]
	# print(metadatadict)






















