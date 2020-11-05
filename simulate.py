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


@njit
def draw_multiple_spheres(canvas, centers, r):
	for center in centers:
		r2 = r * r
		cx, cy, = center
		for i in range(canvas.shape[0]):
			for j in range(canvas.shape[1]):
				if (i - cx)**2 + (j - cy)**2 <= r2:
					canvas[i,j] = 255


def make_random_centers(canvas, n, min_dist=25):
	'''
	Generate random centers of particles

	This is a place holder for bringing in simulated particle trajectories from dynamo
	'''
	x = random.randint(0, canvas_size[0])
	y = random.randint(0, canvas_size[1])
	centers = [(x,y)] # make first particle
	for i in range(n):
		too_close = True
		while too_close:
			x = random.randint(0,canvas_size[0])
			y = random.randint(0,canvas_size[1])
			centers.append((x,y))
			distances = spatial.distance.pdist(centers)
			if all(i > min_dist for i in distances):
				too_close = False
				break
			else:
				centers.pop()
	return centers

def simulate_img2d(canvas_size, centers, r, zoom, gauss):
	'''
	Simulate 2d image of colloids

	centers : list of centers of each particle x,y
	r : radius of particles
	zoom : factor to zoom image out before generation then zoom in as a way to add aliasing
	gauss : kernel for gaussian noise

	TODO fix zooming, use cv2.affinetransform?
	TODO add background
	'''

	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out)
	draw_multiple_spheres(canvas, centers, r)
	canvas = ndimage.zoom(canvas, zoom)
	canvas = ndimage.gaussian_filter(canvas, gauss)
	canvas[canvas<0]=0
	canvas = np.random.poisson(lam=canvas, size=None)
	

	return canvas

if __name__ == "__main__":
	canvas_size = (1024,1024)
	zoom = 0.80
	n=500
	r=20 # provide radius in nm?
	min_dist = 25
	gauss = (11,11)


	centers = make_random_centers(canvas_size, n, min_dist)
	canvas = simulate_img2d(canvas_size, centers, r, zoom, gauss)
	print(np.mean(canvas), np.std(canvas), np.min(canvas))
	plt.imshow(canvas)
	plt.show()


# def make_vid(canvas, centers, frames, r=250, zoom=0.5, gauss = (55,111)):
# 	'''
# 	Generate video on a zero shaped array

# 	TODO unpack centers from trajectories in trackpy format to draw them
# 	For loop should take each frame in turn and redraw particles
# 	'''
# 	vid = []
# 	for i in range(0, 1024, 50):
# 		draw_multiple_spheres(canvas, centers, r)
# 		big_sphere = ndimage.gaussian_filter(canvas, gauss)
# 		vid.append(big_sphere)
# 	vid = np.array(vid, dtype='uint8')

# canvas = np.zeros((2024,2024))
# vid = []
# for i in range(0, 1024, 50):
# 	centers = [[256+i,256+i], [768+i,768+i]]
# 	big_sphere = np.zeros((2024,2024))
# 	draw_multiple_spheres(big_sphere, centers, 256)
# 	big_sphere = ndimage.zoom(big_sphere, 0.5)
# 	big_sphere = ndimage.gaussian_filter(big_sphere, (55,111))
# 	vid.append(big_sphere)
# vid = np.array(vid, dtype='uint8')
# mainViewer(vid)
