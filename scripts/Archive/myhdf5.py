import h5py
import os
import numpy as np

def write_hdf5(dataset, canvas, positions, n, metadata=None):
	path = f'Data/{dataset}.hdf5'
	with h5py.File(path, "w") as f:
		dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype='uint8', data = canvas, compression=1)
		dset.attrs['positions'] = positions

def read_hdf5(dataset, n):
	path = f'Data/{dataset}.hdf5'
	with h5py.File(path, "r") as f:
		print(f.keys())
		canvas = f[str(n)]
		positions = f[str(n)].attrs['positions']
		return np.array(canvas), positions


if __name__ == "__main__":
	write_hdf5('test')