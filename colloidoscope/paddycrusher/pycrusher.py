import cppimport.import_hook
import numpy as np 
import sys
sys.path.append("src/paddycrusher/")
import crusher

def gen_rand_centers(volfrac, canvas_size, diameter = 10):
	centers = crusher.pycrush(volfrac)
	# print(centers.shape, centers.max(), centers.min())
	centers = centers - np.min(centers) # shift positions by left most particle
	centers = centers*diameter # convert from diameter distance to pixel distance
	new_centers = np.array([p for p in centers if p[0]<canvas_size[0] and p[1]<canvas_size[1] and p[2]<canvas_size[2]])
	# print(centers.shape, centers.max(), centers.min())
	return new_centers

if __name__ == '__main__':
	centers = gen_rand_centers(0.35)
	print(centers)
	print(centers.shape, np.nanmin(centers), np.nanmax(centers))

