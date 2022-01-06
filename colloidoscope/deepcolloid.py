import numpy as np
import matplotlib.pyplot as plt
import napari
import h5py
from scipy.spatial.distance import pdist
import cv2
import math
from copy import deepcopy
from .simulator import simulate
import json
from pathlib2 import Path

class DeepColloid:
	def __init__(self, dataset_path) -> None:
		self.dataset_path = dataset_path

	def read_hdf5(self, dataset: str, n: int) -> np.ndarray:
		path = f'{self.dataset_path}/{dataset}.hdf5'
		# print(f'Reading hdf5 dataset: {path} sample number {n}')
		with h5py.File(path, "r") as f:
			canvas = np.array(f[str(n)])
			positions = np.array(f[str(n)+'_positions'])

		json_path = f'{self.dataset_path}/{dataset}.json'
		with open(json_path, "r+") as f:
			json_data = json.load(f)
			metadata = json_data[str(n)]

		return canvas, metadata, positions
	
	def write_hdf5(self, dataset:str, n:int, canvas:np.ndarray,  metadata:dict, positions:np.ndarray, dtype:str='uint8') -> np.ndarray:
		path = f'{self.dataset_path}/{dataset}.hdf5'

		with h5py.File(path, "a") as f:
			dset = f.create_dataset(name=str(n), shape=canvas.shape, dtype=dtype, data = canvas, compression=1)
			dset = f.create_dataset(name=str(n)+'_positions', shape=positions.shape, dtype=dtype, data = positions, compression=1)

		if metadata:
			json_path = f'{self.dataset_path}/{dataset}.json'
			# check if json exists and if not then create it
			json_file = Path(json_path)
			exists = json_file.exists()

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
		path = f'{self.dataset_path}/{dataset}.hdf5'
		with h5py.File(path, "r") as f:
			keys = list(f.keys())
			nums = keys
		# nums = [int(n) for n in list(keys)]
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
		# wrapper for simulator
		return simulate(*args, **kwargs)

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

	def calc_iou_individual(self, center1, center2, diameter,):
		"""Measure intersection over union of two circles with equal diameter
		"""		
		r = diameter / 2
		c = math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2 + (center1[2]-center2[2])**2)
		area = math.pi*r**2
		try:
			q = 2* math.acos(math.radians(c/2*r))
			overlap = (r**2)*(q - math.sin(q)) # area of overlap of circles with same r
		except ValueError:
			overlap = 0
		iou = overlap / (area*2 - overlap)
		return iou

	def get_results(self, gt, pred, diameter, threshold,):

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
		
		gt_idx_thresh = []
		pred_idx_thresh = []
		ious = []
		# for each prediction, measure its iou with ALL ground truths 
		# and append if more than the threshold
		for ipb, pred_pos in enumerate(pred):
			for igb, gt_pos in enumerate(gt):
				iou = self.calc_iou_individual(pred_pos, gt_pos, diameter)
				if iou > threshold:
					gt_idx_thresh.append(igb)
					pred_idx_thresh.append(ipb)
					ious.append(iou)
		ious = np.array(ious)
		# sort by higher iou
		args_desc = np.argsort(ious)[::-1]

		if len(args_desc) == 0:
			# No matches
			tp = 0
			fp = len(pred)
			fn = len(gt)
		else:
			gt_match_idx = []
			pred_match_idx = []
			# in descending iou order check if a match was found
			# select highest iou match for each ground truth
			for idx in args_desc:
				gt_idx = gt_idx_thresh[idx]
				pr_idx = pred_idx_thresh[idx]
				# If the boxes have not been matched, add them to matches
				if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
					gt_match_idx.append(gt_idx)
					pred_match_idx.append(pr_idx)

			tp = len(gt_match_idx)
			fp = len(pred) - len(pred_match_idx)
			fn = len(gt) - len(gt_match_idx)

		return tp, fp, fn

	def get_precision_recall(self, ground_truths, predictions, diameter, threshold):


		# take block of different images and find result
		tp, fp, fn = self.get_results(ground_truths, predictions, diameter, threshold,)

		try:
			precision = tp/(tp + fp)
		except ZeroDivisionError:
			precision = 0.0
		try:
			recall = tp/(tp + fn)
		except ZeroDivisionError:
			recall = 0.0
		precision = np.array(precision)
		recall = np.array(recall)

		return precision, recall

	def average_precision(self, ground_truth, prediction, diameter=10):
		print('Calculating average precision, threshold:')

		average_precisions = []
		precisions = []
		recalls = []
		thresholds = np.linspace(0,1,30)
		for thresh in thresholds:
			print(thresh)
			prec, rec = self.get_precision_recall(ground_truth, prediction, diameter, thresh,)
			# precisions = precisions + prec
			# recalls = recalls + rec
			precisions.append(prec)
			recalls.append(rec)
		precisions = np.array(precisions)
		recalls = np.array(recalls)

		prec_at_rec = []
		for recall_level in np.linspace(0.0, 1.0, 11):
			try:
				args = np.argwhere(recalls >= recall_level).flatten()
				prec = max(precisions[args])
			except ValueError:
				prec = 0.0
			prec_at_rec.append(prec)
		average_precision = np.mean(prec_at_rec)

		return average_precision, precisions, recalls

	def crop3d(self, array, roiSize, center=None):
		roiZ, roiY, roiX = roiSize
		zl = int(roiZ / 2)
		yl = int(roiY / 2)
		xl = int(roiX / 2)

		if center == None:
			c = int(array.shape[0] / 2)
			center = [c, c, c]

		z, y, x = center
		z, y, x = int(z), int(y), int(x)
		array = array[z - zl : z + zl, y - yl : y + yl, x - xl : x + xl]
		return array



