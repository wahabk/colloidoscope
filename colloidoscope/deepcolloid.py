import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import napari
import h5py
from scipy.spatial.distance import pdist, cdist
from .simulator import simulate
from .explore_lif import Reader
from .predict import detect
import json
from pathlib2 import Path
from skimage import io
from tqdm import tqdm
import trackpy as tp
import math
from scipy.signal import convolve2d

class DeepColloid:
	def __init__(self, dataset_path=None) -> None:
		if dataset_path == None: 
			self.dataset_initialised = False
		else:
			self.dataset_initialised = True
			self.dataset_path = dataset_path

	def detect(self, *args, **kwargs):
		"""Detect 3d spheres from confocal microscopy

		Args:
			array (np.ndarray): Image for particles to be detected from.
			diameter (Union[int, list], optional): Diameter of particles to feed to TrackPy, can be int or list the same length as image dimensions. Defaults to 5. If post_processing == str(max) this has to be int it will be min_distance.
			model (torch.nn.Module, optional): Pytorch model. Defaults to None.
			weights_path (Union[str, Path], optional): Path to model weights file. Defaults to None.
			patch_overlap (tuple, optional): Overlap for patch based inference, overlap must be diff between input and output shape (if they are not the same). Defaults to (16, 16, 16).
			roiSize (tuple, optional): Size of ROI for model. Defaults to (64,64,64).
			debug (bool, optional): Option to return model output and positions in format for testing. Defaults to False.
			post_processing (str, optional): one of ["tp", "max", "log"]
			run_on (str, optional): Which device to run on, can be "cpu" or "cuda"

		Returns:
			pd.DataFrame: TrackPy positions dataframe
		"""	
		
		return detect(*args, **kwargs)

	def read_tif(self, path):
		return io.imread(path)

	def Explore_lif_reader(self, *args, **kwargs):
		return Reader(*args, **kwargs)

	def read_hdf5(self, dataset: str, n: int, read_labels=True, read_diameters=True, read_positions=True) -> dict:
		"""This reads simulated data stored in hdf5

		note the first index is 1 not 0 unlike gsd

		Args:
			dataset (str): _description_
			n (int): _description_

		Returns:
			dict: _description_
		"""
		if self.dataset_initialised == False:
			raise Exception('Dataset not initialised')
		path = f'{self.dataset_path}/{dataset}.hdf5'
		# print(f'Reading hdf5 dataset: {path} sample number {n}')
		data = {}
		with h5py.File(path, "r") as f:
			data['image'] = np.array(f[str(n)])
			if read_labels:		data['label'] = np.array(f[str(n)+'_labels'])
			if read_positions:	data['positions'] = np.array(f[str(n)+'_positions'])
			if read_diameters:	data['diameters'] = np.array(f[str(n)+'_diameters'])

		
		json_path = f'{self.dataset_path}/{dataset}.json'
		with open(json_path, "r+") as f:
			json_data = json.load(f)
			metadata = json_data[str(n)]

		data['metadata'] = metadata

		# data = {
		# 	'image' : canvas,
		# 	'positions' : positions,
		# 	'label' : label,
		# 	'diameters' : diameters,
		# 	'metadata' : metadata,
		# }

		return data

	def read_metadata(self, dataset: str, n: int) -> dict:
		if self.dataset_initialised == False:
			raise Exception('Dataset not initialised')
		json_path = f'{self.dataset_path}/{dataset}.json'
		with open(json_path, "r+") as f:
			json_data = json.load(f)
			metadata = json_data[str(n)]

		path = f'{self.dataset_path}/{dataset}.hdf5'
		with h5py.File(path, "r") as f:
			positions = np.array(f[str(n)+'_positions'])
			diameters = np.array(f[str(n)+'_diameters'])

		return metadata, positions, diameters
	
	def write_hdf5(self, dataset:str, n:int, canvas:np.ndarray, metadata:dict=None, positions:np.ndarray=None, label:np.ndarray=None, diameters=None, dtype:str='uint8') -> np.ndarray:
		if self.dataset_initialised == False:
			raise Exception('Dataset not initialised')
		path = f'{self.dataset_path}/{dataset}.hdf5'

		with h5py.File(path, "a") as f:
			dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype=dtype, data = canvas, compression=1)
			# TODO change to csv
			if isinstance(label, np.ndarray): 		dset = f.create_dataset(name=str(n)+'_labels', shape=label.shape, dtype='float32', data = label, compression=1)
			if isinstance(positions, np.ndarray): 	dset = f.create_dataset(name=str(n)+'_positions', shape=positions.shape, dtype='float32', data = positions, compression=1)
			if isinstance(diameters, np.ndarray): 	dset = f.create_dataset(name=str(n)+'_diameters', shape=diameters.shape, dtype='float32', data = diameters, compression=1)

		if metadata:
			# TODO change to csv
			json_path = f'{self.dataset_path}/{dataset}.json'
			# check if json exists and if not then create it
			json_file = Path(json_path)
			exists = json_file.exists()

			# check if exists or create new one
			if exists:
				with open(json_path, 'r') as json_file:
					json_data = json.load(json_file)
			else:
				json_data = {}
			json_data[str(n)] = metadata
			with open(json_path, 'w') as json_file:			
				json.dump(json_data, json_file, sort_keys=True, indent=4)

		return

	def get_hdf5_keys(self, dataset) -> list:
		if self.dataset_initialised == False:
			raise Exception('Dataset not initialised')
		path = f'{self.dataset_path}/{dataset}.hdf5'
		with h5py.File(path, "r") as f:
			keys = list(f.keys())
			nums = keys
		# nums = [int(n) for n in list(keys)]
		return nums

	def view(self, array:np.ndarray, positions:np.ndarray=None, true_positions:np.ndarray=None, label:np.ndarray=None) -> None:

		viewer = napari.view_image(array, name='Scan')
		
		if label is not None:
			diff = [(a-b)/2 for a, b in zip(array.shape, label.shape)]
			# print(diff)
			viewer.add_image(label*255, opacity=0.25, colormap='red', name='label')#, translate=diff)	
		
		if positions is not None:
			if label is None: diff=[0 for i in array.shape]
			# array = np.array([np.stack((img,)*3, axis=-1) for img in array])
			# array = self.label_scan(array, positions)
			viewer.add_points(positions, n_dimensional=True, size=5, symbol="x",  edge_color=[1,0,0], face_color=[1,0,0])#, translate=diff)

		if true_positions is not None:
			if label is None: diff=[0 for i in array.shape]
			# array = np.array([np.stack((img,)*3, axis=-1) for img in array])
			# array = self.label_scan(array, true_positions)
			viewer.add_points(true_positions, n_dimensional=True, size=5, symbol="o", edge_color=[0,1,0], face_color=[0,1,0])#, translate=diff)


		napari.run()

	def simulate(self, *args, **kwargs):
		# wrapper for simulator
		return simulate(*args, **kwargs)

	def vol_frac(self, centers, r, canvas_size):
		vol = (4/3)*  np.pi * r**3
		num = len(centers)
		z, x, y = canvas_size
		volsys = z*x*y
		volfrac = (vol * num) / volsys
		return volfrac

	def round_up_to_odd(self, f):
		return int(np.ceil(f) // 2 * 2 + 1) # // is floor division

	def get_gr(self, positions, cutoff, bins, minimum_gas_number=1e4):
		# from yushi yang
		#TODO add multiframe

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

		# print(hist, rg_hist)
		# rg_hist[rg_hist==0] = 0.000001

		epsilon = 1e-10
		hist = hist / rg_hist # pdfs
		hist[np.isnan(hist)] = epsilon # fix division by zero
		bin_centres = (bins[1:] + bins[:-1]) / 2
		return bin_centres, hist # as x, y

	def plot_gr(self, x, y, diameter, label=f'prediction', color='gray', axs=None, fontsize='medium', xlim=(0,5)):
		if isinstance(diameter, list): diameter = min(diameter)
		x = x / diameter
		if axs is None:
			plt.plot(x, y, label=label, color=color)
			plt.xlabel("$r / \sigma$", fontsize=fontsize)
			plt.ylabel("$g(r)$", fontsize=fontsize)
			plt.xticks(list(range(xlim[0], xlim[1])))
			plt.xlim(xlim)
			plt.legend()
		else:
			axs.plot(x, y, label=label, color=color)
			axs.set_xlabel("$r / \sigma$", fontsize=fontsize)
			axs.set_ylabel("$g(r)$", fontsize=fontsize)
			axs.set_xticks(list(range(xlim[0], xlim[1])))
			axs.set_xlim(xlim)
		return

	@staticmethod
	def _calc_iou_dist(distances, diameters, threshold):
		"""
		Measure distance between two spheres normalised by diameters
		"""

		gt_idx_thresh = []
		pred_idx_thresh = []
		ious = []
		for ipb, pred_list in enumerate(distances):
			for igb, dist in enumerate(pred_list):
				diameter = diameters[igb]

				iou_dist = 1 - (dist/diameter)
				if iou_dist < 0 : iou_dist = 0
				iou = iou_dist
				if iou > threshold:
					gt_idx_thresh.append(igb)
					pred_idx_thresh.append(ipb)
					ious.append(iou)
		return gt_idx_thresh, pred_idx_thresh, ious

	
	
	def _get_results(self, gt, pred, diameters, threshold,) -> tuple:
		from numba import njit

		tp, fp, fn = 0, 0, 0

		if len(pred) == 0:
			tp = 0
			fp = 0
			fn = len(gt)
			return tp, fp, fn
		if len(gt) == 0:
			tp = 0
			fp = len(pred)
			fn = 0
			return tp, fp, fn
		
		# for each prediction, measure its iou with ALL ground truths 
		# and append if more than the threshold
		dists = cdist(pred, gt)
		jitted_calc_iou_dist = njit(self._calc_iou_dist)
		gt_idx_thresh, pred_idx_thresh, ious = jitted_calc_iou_dist(dists, diameters, threshold)
		ious = np.array(ious)

		# sort by higher iou
		args_desc = np.argsort(ious)
		args_desc = args_desc[::-1]

		if len(args_desc) == 0:
			# No matches
			tp = 0
			fp = len(pred)
			fn = len(gt)
		else:
			iou_at_pred = []
			gt_match_idx = []
			pred_match_idx = []
			iou_index = []
			# in descending iou order check if a match was found
			# select highest iou match for each ground truth
			for i, idx in enumerate(args_desc):
				gt_idx = gt_idx_thresh[idx]
				pr_idx = pred_idx_thresh[idx]
				# If the boxes have not been matched, add them to matches
				if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
					gt_match_idx.append(gt_idx)
					pred_match_idx.append(pr_idx)
					iou_index.append(i)
					iou_at_pred.append(ious[idx])
			
			tp = len(gt_match_idx) # ones that are in both
			fp = len(pred) - len(pred_match_idx) # predictions not matched
			fn = len(gt) - len(gt_match_idx) # grount truths not in pred

			# import pdb; pdb.set_trace()

		return tp, fp, fn

	def get_precision_recall(self, ground_truths, predictions, diameters, threshold):
		
		# take block of different images and find result
		tp, fp, fn = self._get_results(ground_truths, predictions, diameters, threshold)

		try:
			precision = tp/(tp + fp)
		except ZeroDivisionError:
			precision = 0.0
		try:
			recall = tp/(tp + fn)
		except ZeroDivisionError:
			recall = 0.0
		precision = precision
		recall = recall

		return precision, recall

	def average_precision(self, ground_truth, prediction, diameters):

		# based on https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734


		print('Calculating average precision, threshold:')

		precisions = []
		recalls = []
		predictions = []
		thresholds = np.linspace(0,1,10)
		for thresh in tqdm(thresholds):
			prec, rec = self.get_precision_recall(ground_truth, prediction, diameters, thresh,)
			precisions.append(prec)
			recalls.append(rec)
			# predictions.append(pred) # predictions are here for testing sklearn.metrics from_estimator
		print('Done')

		precisions = np.array(precisions)
		recalls = np.array(recalls)
		recalls = np.flip(recalls) # TODO why do i need to flip ????

		ap = np.trapz(precisions, x=recalls) # integrate

		return ap, precisions, recalls, thresholds

	def plot_pr(self, ap, precisions, recalls, thresholds, name, tag='o-', axs=None, title='Average Precision', *args, **kwargs):
		# display = metrics.PrecisionRecallDisplay(precision=precisions, 
		# recall=recalls, estimator_name=name).plot()

		if axs is None:
			plt.plot(recalls, precisions, tag, label=f'{name} AP={ap:.2f}', *args, **kwargs)
			if title is None: pass
			else: plt.title(title)
			plt.xlabel("Recall")
			plt.ylabel("Recall")
			plt.xlim([-0.1,1.1])
			plt.ylim([-0.1,1.1])
			plt.legend()
			return plt.gcf()

		else:
			axs.plot(recalls, precisions, tag, label=f'{name} AP={ap:.2f}', *args, **kwargs)
			if title is None: pass
			else: axs.set_title(title)
			axs.set_xlabel("Recall")
			axs.set_ylabel("Recall")
			axs.set_xlim([-0.1,1.1])
			axs.set_ylim([-0.1,1.1])
			axs.legend()

	def run_trackpy(self, array, diameter=5, *args, **kwargs) -> np.ndarray: #, pd.DataFrame]
		df = tp.locate(array, diameter=diameter, *args, **kwargs)
		f = list(zip(df['z'], df['y'], df['x']))
		tp_predictions = np.array(f, dtype='float32')

		return tp_predictions, df

	def crop3d(self, array, roiSize, center=None):
		roiZ, roiY, roiX = roiSize
		zl = int(roiZ / 2)
		yl = int(roiY / 2)
		xl = int(roiX / 2)

		if center == None:
			center = [int(array.shape[0] / 2), int(array.shape[1] / 2), int(array.shape[2] / 2)]

		z, y, x = center
		z, y, x = int(z), int(y), int(x)
		array = array[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl]
		return array

	def estimate_noise(self, I):

		H, W = I.shape

		M = [[1, -2, 1],
			[-2, 4, -2],
			[1, -2, 1]]

		sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
		sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

		return sigma
