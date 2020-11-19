from numba import njit
import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from mainviewer import mainViewer
from concurrent.futures import ProcessPoolExecutor
import random
from math import sqrt
import skimage
#255x255x128



def draw_sphere(canvas, center, r):
	cz, cy, cx = center
	r2 = r**2
	for i in range(canvas.shape[0]):
			for j in range(canvas.shape[1]):
				for k in range(canvas.shape[2]):
					if (i - cz)**2 + (j - cx)**2 + (k - cy)**2 <= r2:
						canvas[i,j,k] = 255
	return canvas

@njit
def draw_multiple_spheres(canvas, centers, r):
	for center in centers:
		cz, cy, cx = center
		r2 = r**2
		for i in range(canvas.shape[0]):
				for j in range(canvas.shape[1]):
					for k in range(canvas.shape[2]):
						if (i - cz)**2 + (j - cx)**2 + (k - cy)**2 <= r2:
							canvas[i,j,k] = 255
	return canvas

def make_random_centers_3d(canvas_size, n, zoom, min_dist):
	'''
	Generate random centers of particles

	This is a place holder for bringing in simulated particle trajectories from dynamo
	'''
	canvas_size = [int(c/zoom) for c in canvas_size]
	min_dist = min_dist/zoom
	z = random.randint(0, canvas_size[0])
	y = random.randint(0, canvas_size[1])
	x = random.randint(0, canvas_size[2])
	centers = [(z,y,x)] # make first particle
	for i in range(n):
		too_close = True
		while too_close:
			z = random.randint(0, canvas_size[0])
			y = random.randint(0, canvas_size[1])
			x = random.randint(0, canvas_size[2])
			centers.append((z,y,x))
			distances = spatial.distance.pdist(centers)
			if all(i > min_dist for i in distances):
				too_close = False
				break
			else:
				centers.pop() # get rid of last element if too close
	return centers

def simulate_img3d(canvas_size, centers, r, zoom, gauss):
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

	draw_multiple_spheres(canvas, centers, r)

	canvas = ndimage.zoom(canvas, zoom)
	canvas = ndimage.gaussian_filter(canvas, gauss)
	canvas[canvas<0]=0
	# canvas = np.random.poisson(lam=canvas, size=None)
	PEAK = 0.1
	canvas[canvas<0]=0
	# poisson_noise = np.sqrt(canvas) * np.random.normal(0, 3, canvas.shape)
	# canvas = canvas + poisson_noise
	canvas[canvas<0]=0
	#canvas[canvas>10]=255

	return canvas



if __name__ == "__main__":
	canvas_size = (100,512,512)
	r = 30
	zoom = 0.5
	gauss = (9,9,11)
	n = 25
	min_dist = 40


	centers = make_random_centers_3d(canvas_size, n, zoom, min_dist)
	canvas = simulate_img3d(canvas_size, centers, r, zoom, gauss)
	mainViewer(canvas)
	print(np.mean(canvas), np.std(canvas), np.min(canvas), np.max(canvas))
