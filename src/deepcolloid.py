import numpy as np
import matplotlib.pyplot as plt
import random
import napari
import h5py
from scipy.spatial.distance import pdist


class DeepColloid:
	def __init__(self) -> None:
		pass


	def read_hdf5(self, dataset: str, n: int, positions: bool =False,) -> None:
		path = f'Data/{dataset}.hdf5'
		with h5py.File(path, "r") as f:
			canvas = f[str(n)]
			if positions: 
				positions = f[str(n)].attrs['positions']
				return np.array(canvas), np.array(positions)
			else: 
				return np.array(canvas)		
	
	def write_hdf5(self, dataset: np.ndarray, n: int, canvas: np.ndarray, positions: bool=False, metadata: bool=None,) -> np.ndarray:
		path = f'Data/{dataset}.hdf5'
		with h5py.File(path, "a") as f:
			dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype='uint8', data = canvas, compression=1)
			if positions: dset.attrs['positions'] = positions

	def vol_frac(self, centers, r, canvas_size):
		vol = (4/3)*  np.pi * r**3
		num = len(centers)
		z, x, y = canvas_size
		volsys = z*x*y
		volfrac = (vol * num) / volsys
		return volfrac

	def view(self, array):
		pass

	def simulate(self, array):
		pass

	def get_gr(self, positions, cutoff, bins, minimum_gas_number=1e4):
		# from yushi yang
		bins = np.linspace(0, cutoff, bins)
		drs = bins[1:] - bins[:-1]
		distances = pdist(positions).ravel()

		if positions.shape[0] < minimum_gas_number:
			rg_hists = []
			for i in range(int(minimum_gas_number) // positions.shape[0] + 2):
				random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
				rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]
				rg_hists.append(rg_hist)
			rg_hist = np.mean(rg_hists, 0)

		else:
			random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
			rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]

		hist = np.histogram(distances, bins=bins)[0]
		hist = hist / rg_hist # pdfs
		hist[np.isnan(hist)] = 0
		bin_centres = (bins[1:] + bins[:-1]) / 2
		return bin_centres, hist # as x, y

		