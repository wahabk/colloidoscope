from make_dataset import *
import numpy as np
import cv2
from skimage.util.shape import view_as_windows
from skimage import io
from pathlib2 import Path
import math

def make_blocks(stack,shape):
	'''
	add overlap?
	stack should be a yXmXn matrix, and t should even divides m,n
	returns a list of 3D blocks of size yXtXt

	from https://stackoverflow.com/questions/39429900/split-a-3d-numpy-array-into-3d-blocks

	'''
	z_length,length = shape
	z = range(0,stack.shape[0],z_length)
	y = range(0,stack.shape[1],length)
	x = range(0,stack.shape[2],length)
	reshaped = []
	for i in z:
		for j in y:
			for k in x:
				reshaped.append(stack[i:i+z_length:,j:j+length,k:k+length])
	reshaped = np.array(reshaped)
	return reshaped

def get_tiles_from_dir(dataPath, tile_size):
	all_tiles = []
	for stackPath in dataPath.iterdir():
		#print(stackPath)
		stack = io.imread(str(stackPath))
		if len(stack.shape) == 4:
			stacks = stack
			for stack in stacks:
				tiles = make_blocks(stack, (tile_size[0],tile_size[1]))
				for t in tiles:
					all_tiles.append(t)
		
		elif len(stack.shape) == 3:
			z, y, x = stack.shape
			if (z % 32) != 0:
				factor = math.floor(z/32)
				stack = stack[0:(factor*32)]
			tiles = make_blocks(stack, (tile_size[0],tile_size[1]))
			for t in tiles:
				all_tiles.append(t)
	all_tiles = np.array(all_tiles)
	return all_tiles

def unmake_blocks(x,d,m,n):
	'''
	TODO make this work in 3d
	this takes a list of matrix blocks of size dXd that is m*n/d^2 long 
	returns a 2D array of size mXn
	
	
	from https://stackoverflow.com/questions/39429900/split-a-3d-numpy-array-into-3d-blocks
	'''
	rows = []
	for i in range(0,int(m/d)):
		rows.append(np.hstack(x[i*int(n/d):(i+1)*int(n/d)]))
	return np.vstack(rows)

if __name__ == "__main__":
	canvas_size=(32,128,128)
	dataset = 'Real'
	dataPath = Path('Data/Tracking/Ready/')

	tiles = get_tiles_from_dir(dataPath, tile_size=canvas_size)

	for i, canvas in enumerate(tiles):
		print(i)
		write_hdf5(dataset, i, canvas)

	# for i, canvas in enumerate(tiles[0:6]):
	# 	print(i)
	# 	make_gif(canvas, f'output/Example/{i}_real_scan.gif', fps = 7)
