import numpy as np
import matplotlib.pyplot as plt
import napari
import h5py
from scipy.spatial.distance import pdist
from moviepy.editor import ImageSequenceClip
import cv2
import math
from copy import deepcopy
from .simulator import *


class DeepColloid:
	def __init__(self, dataset_path) -> None:
		self.dataset_path = dataset_path

	def read_hdf5(self, dataset: str, n: int, return_positions: bool=False,) -> np.ndarray:
		path = f'{self.dataset_path}/{dataset}.hdf5'
		# print(f'Reading hdf5 dataset: {path} sample number {n}')
		with h5py.File(path, "r") as f:
			canvas = f[str(n)]
			if return_positions: 
				positions = f[str(n)].attrs['positions']
				return np.array(canvas), np.array(positions)
			else: 
				return np.array(canvas)		
	
	def write_hdf5(self, dataset: np.ndarray, n: int, canvas: np.ndarray, positions: np.ndarray=False, metadata: bool=None,) -> np.ndarray:
		path = f'{self.dataset_path}/{dataset}.hdf5'
		with h5py.File(path, "a") as f:
			dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype='uint8', data = canvas, compression=1)
			if positions is not None: dset.attrs['positions'] = positions
		return

	def get_hdf5_keys(self, dataset) -> list:
		path = f'{self.dataset_path}/{dataset}.hdf5'
		with h5py.File(path, "r") as f:
			keys = list(f.keys())
		nums = [int(n) for n in list(keys)]
		return nums

	def view(self, array:np.ndarray, positions:np.ndarray=None) -> None:
		if positions is not None:
			array = np.array([np.stack((img,)*3, axis=-1) for img in array])
			array = self.label_scan(array, positions)

		napari.view_image(array)

	def label_scan(self, array: np.ndarray, positions: list) -> np.ndarray:
		canvas = deepcopy(array)
		#decompose grayscale numpy array into RGB
		
		for z, y, x in positions:
			z, y, x = math.floor(z), int(y), int(x)
			cv2.rectangle(canvas[z], (x - 1, y - 1), (x + 1, y + 1), (250,0,0), -1)
			cv2.circle(canvas[z], (x, y), 5, (0, 250, 0), 1)
		
		return canvas

	def simulate(self, *args, **kwargs):
		return simulate_img3d(*args, **kwargs)

	def make_gif(self, canvas, file_name, fps = 7, positions=[], scale=None):
		#decompose grayscale numpy array into RGB
		if np.max(canvas) < 2: canvas *= 255
		new_canvas = np.array([np.stack((img,)*3, axis=-1) for img in canvas])

		print(np.shape(new_canvas), type(new_canvas))

		for z, y, x in positions:
			z, y, x = math.floor(z), int(y), int(x)
			if z==31:z=30
			cv2.rectangle(new_canvas[z], (x - 1, y - 1), (x + 1, y + 1), (250,0,0), -1)
			# cv2.circle(new_canvas[z], (x, y), 5, (0, 250, 0), 1)
		
		print(np.shape(new_canvas), type(new_canvas))

		# if positions:
		# 	new_canvas = self.label_scan(new_canvas, positions)

		if scale is not None:
			im = new_canvas[0]
			width = int(im.shape[1] * scale / 100)
			height = int(im.shape[0] * scale / 100)
			dim = (width, height)

			# resize image
			resized = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in new_canvas]
			new_canvas = resized
	
		new_canvas = list(new_canvas)
		print(np.shape(new_canvas), type(new_canvas))
		clip = ImageSequenceClip(new_canvas, fps=fps)
		clip.write_gif(file_name, fps=fps)

	def vol_frac(self, centers, r, canvas_size):
		vol = (4/3)*  np.pi * r**3
		num = len(centers)
		z, x, y = canvas_size
		volsys = z*x*y
		volfrac = (vol * num) / volsys
		return volfrac

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

