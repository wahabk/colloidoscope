from numba import njit
import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from mainviewer import mainViewer
from concurrent.futures import ProcessPoolExecutor
import random
from math import sqrt
import skimage
from poisson_disc_sampling import poisson_disc_sampling
import os
#255x255x128
print(os.cpu_count())
from concurrent.futures import ProcessPoolExecutor
from skimage.util import random_noise

def draw_sphere(canvas, center, r):
	cz, cy, cx = center
	for i in range(canvas.shape[0]):
			for j in range(canvas.shape[1]):
				for k in range(canvas.shape[2]):
					if (i - cz)**2 + (j - cy)**2 + (k - cx)**2 <= r**2:
						canvas[i,j,k] = 255
	return canvas


def draw_multiple_spheres(canvas, centers, r):
	for center in centers:
		cz, cy, cx = center
		for i in range(canvas.shape[0]):
				for j in range(canvas.shape[1]):
					for k in range(canvas.shape[2]):
						if (i - cz)**2 + (j - cy)**2 + (k - cx)**2 <= r**2:
							canvas[i,j,k] = 255
	return canvas



@njit
def draw_slice(a):
	s, z, centers, r = a
	for i in range(s.shape[0]):
		for j in range(s.shape[1]):
			for center in centers:
				cz, cy, cx = center
				if (z - cz)**2 + (i - cy)**2 + (j - cx)**2 <= r**2:
					s[i,j] = 255
	return s

def draw_spheres_sliced(canvas, centers, r):
	new_canvas = []
	args = [(s, z, centers, r) for z, s in enumerate(canvas)]

	with ProcessPoolExecutor(max_workers=10) as pool:
		new_canvas = pool.map(draw_slice, args)
	canvas = list(new_canvas)

	canvas = np.array(canvas)
	return canvas



def simulate_img3d(canvas_size, r, min_dist, zoom, gauss, k=30):
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
	centers = poisson_disc_sampling(min_dist, np.array(zoom_out), k)
	print('centers shape =', len(centers))

	new_canvas = draw_spheres_sliced(canvas, centers, r)

	print('done drawing')
	canvas = new_canvas
	print(canvas.shape)
	canvas = ndimage.zoom(canvas, zoom)
	canvas = ndimage.gaussian_filter(canvas, gauss)
	
	canvas = np.array(canvas, dtype='uint8')
	canvas[canvas<0]=0
	return canvas



if __name__ == "__main__":
	canvas_size = (50,512,512)
	r = 10
	zoom = 0.8
	gauss = (3,3,3)
	n = 25
	min_dist = 20

	

	canvas = simulate_img3d(canvas_size, r, min_dist, zoom, gauss)
	mainViewer(canvas)
	print(np.mean(canvas), np.std(canvas), np.min(canvas), np.max(canvas))























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

