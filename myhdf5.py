from pathlib2 import Path
import h5py
import os
import numpy as np

def write_hdf5(dataset, canvas, positions, n, metadata=None):
	path = Path(f'Data/{dataset}.hdf5')
	if path.exists() == False: 
		f = h5py.File(path, "w")
		dset = f.create_dataset('samples', data = canvas)
		dset.attrs['positions'] = positions
		f.close()
	else:
		with h5py.File(path, "w") as f:
			sofar = [int(k) for k in f.keys()]
			print(sofar)
			this_sample = str(max(sofar)+1)
			dset = f.create_dataset(this_sample, data = canvas)
			dset.attrs['positions'] = positions


def read_hdf5(dataset, n):
	path = Path(f'Data/{dataset}.hdf5')
	if path.exists():
		with h5py.File(path, "r") as f:
			print(f.keys())
			canvas = f[str(n)]
			positions = f[str(n)].attrs['positions']
			return np.array(canvas), positions


if __name__ == "__main__":
	write_hdf5('test')