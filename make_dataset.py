from sim3d import simulate_img3d
import h5py
from varname import nameof
from mainviewer import mainViewer
import numpy as np

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


if __name__ == "__main__":
	canvas_size=(32,256,256)
	r = 10
	zoom = 0.75
	gauss = (7,3,3)
	min_dist = 2*r
	k = 30
	dataset = 'Test'

	for n in range(1,31):
		print(n)
		canvas, positions, label = simulate_img3d(canvas_size, r, min_dist, zoom, gauss, k=30)
		write_hdf5(dataset, n, canvas, positions)
		write_hdf5(dataset+'_labels', n, label)
		
	for n in range(1,3):
		canvas, positions = read_hdf5(dataset, n, positions=True)
		label = read_hdf5(dataset+'_labels', n)
		mainViewer(canvas)
		mainViewer(label)
		print(len(positions), positions)
		print(canvas.shape, label.shape)