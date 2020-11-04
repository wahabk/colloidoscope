from numba import njit
import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from mainviewer import mainViewer

#255x255x128

@njit
def draw_sphere(canvas, center, r):
	r2 = r * r
	cx, cy, = center
	for i in range(canvas.shape[0]):
		for j in range(canvas.shape[1]):
			if (i - cx)**2 + (j - cy)**2 <= r2:
				canvas[i,j] = 255

def draw_multiple_spheres(canvas, centers, r):
	for center in centers:
		draw_sphere(canvas, center, r)

def make_vid(canvas, centers, frames, r=250, zoom=0.5, gauss = (55,111)):
	'''
	Generate video on a zero shaped array

	TODO unpack centers from trajectories in trackpy format to draw them
	For loop should take each frame in turn and redraw particles
	'''
	vid = []
	for i in range(0, 1024, 50):
		draw_multiple_spheres(canvas, centers, r)
		big_sphere = ndimage.gaussian_filter(canvas, gauss)
		vid.append(big_sphere)
	vid = np.array(vid, dtype='uint8')

def make_random_centers(canvas_size, n, min_dist=10):
	pass

def simulate_img2d(centers)
	canvas_size = (512,512)
	r=10 # provide radius in nm?
	zoom=0.5
	gauss = (55,111)

	
	zoom_out = [int(c/zoom) for c in canvas_size]
	canvas = np.zeros(zoom_out)

	draw_multiple_spheres(canvas, centers, r)
	canvas = ndimage.zoom(big_sphere, zoom)
	big_sphere = ndimage.gaussian_filter(big_sphere, (55,111))

	# TODO add noise and background
	return

if __name__ == "__main__":





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
