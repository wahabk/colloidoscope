from sim3d import simulate_img3d
import h5py
from mainviewer import mainViewer
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip
from random import randrange, uniform
import math
from skimage.util.shape import view_as_blocks
from skimage import io

def write_hdf5(dataset, n, canvas, positions=False, metadata=None):
	path = f'Data/{dataset}.hdf5'
	with h5py.File(path, "a") as f:
		dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype='uint8', data = canvas, compression=1)
		if positions: dset.attrs['positions'] = positions

def read_hdf5(dataset, n, positions=False):
	path = f'Data/{dataset}.hdf5'
	with h5py.File(path, "r") as f:
		canvas = f[str(n)]
		if positions: 
			positions = f[str(n)].attrs['positions']
			return np.array(canvas), np.array(positions)
		else: 
			return np.array(canvas)				

def make_gif(canvas, file_name, fps = 7, positions=None, scale=None):
	#decompose grayscale numpy array into RGB
	new_canvas = np.array([np.stack((img,)*3, axis=-1) for img in canvas])

	if positions is not None:
		for z, y, x in positions:
			z, y, x = math.floor(z), int(y), int(x)
			if z==31:z=30
			cv2.rectangle(new_canvas[z], (x - 1, y - 1), (x + 1, y + 1), (250,0,0), -1)
			# cv2.circle(new_canvas[z], (x, y), 5, (0, 250, 0), 1)

	if scale is not None:
		im = new_canvas[0]
		width = int(im.shape[1] * scale / 100)
		height = int(im.shape[0] * scale / 100)
		dim = (width, height)

		# resize image
		resized = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in new_canvas]
		new_canvas = resized
 

	# write_gif(new_canvas, file_name, fps = fps)
	
	clip = ImageSequenceClip(list(new_canvas), fps=fps)
	clip.write_gif(file_name, fps=fps)
	

if __name__ == "__main__":
	canvas_size=(32,128,128)
	
	dataset = 'TF'
	n_samples = 2

	# for n in range(1,n_samples+1):
	# 	print(f'{n}/{n_samples}')
	# 	zoom = 0.75
	# 	xykernel = randrange(1,4,2)
		
	# 	gauss = (randrange(5,8,2),xykernel,xykernel)
	# 	gauss = (5,1,1)
	# 	brightness = randrange(180,250)
	# 	noise = uniform(0.002, 0.008)
	# 	canvas, positions, label = simulate_img3d(canvas_size, zoom, gauss, noise=noise, volfrac = 0.3)
		
	# 	# mainViewer(canvas, positions=positions)
	# 	# mainViewer(label, positions=positions)
	# 	# write_hdf5(dataset, n, canvas, positions)
	# 	# write_hdf5(dataset+'_labels', n, label)
	# 	canvas, positions, label = None, None, None
		
	for n in range(1,4):
		canvas, positions = read_hdf5(dataset, n, positions=True)
		label = read_hdf5(dataset+'_labels', n)
		make_gif(canvas, f'output/Example/{dataset}_scan_{n}.gif', fps = 7, scale=300)
		make_gif(label, f'output/Example/{dataset}_scan_{n}labels.gif', fps = 7, scale=300)

