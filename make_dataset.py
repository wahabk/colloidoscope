from sim3d import simulate_img3d
import h5py
from mainviewer import mainViewer
import numpy as np
from array2gif import write_gif
import cv2
from moviepy.editor import ImageSequenceClip

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

def make_gif(canvas, file_name, fps = 7, positions=None, scale=300):
	#decompose grayscale numpy array into RGB
	new_canvas = np.array([np.stack((img,)*3, axis=-1) for img in canvas])

	if positions is not None:
		for z, y, x in positions:
			z, y, x = int(z), int(y), int(x)
			cv2.rectangle(new_canvas[z], (x - 2, y - 2), (x + 2, y + 2), (250,0,0), -1)
			cv2.circle(new_canvas[z], (x, y), 10, (0, 250, 0), 2)

	
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
	r = 10
	zoom = 0.75
	gauss = (7,3,3)
	min_dist = 2*r+1
	k = 100
	dataset = 'First'

	# for n in range(1,101):
	# 	print(n)
	# 	canvas, positions, label = simulate_img3d(canvas_size, r, min_dist, zoom, gauss, k=k)
	# 	write_hdf5(dataset, n, canvas, positions)
	# 	write_hdf5(dataset+'_labels', n, label)
		
	for n in range(1,2):
		canvas, positions = read_hdf5(dataset, n, positions=True)
		label = read_hdf5(dataset+'_labels', n)
		make_gif(canvas, 'scan.gif', fps = 7, positions=positions, scale=300)
		make_gif(label, 'scan_labels.gif', fps = 7, positions=positions, scale=300)

		# mainViewer(canvas)
		# mainViewer(label)
		# print(len(positions), positions)
		# print(canvas.shape, label.shape)