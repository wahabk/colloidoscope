import numpy as np
import matplotlib.pyplot as plt
import napari
import h5py
from scipy.spatial.distance import pdist
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

	def average_precision(self, ground_truths, predictions, diameter=10):

		def calc_iou_individual(center1, center2, diameter,):
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

		def get_results(gt, pred, diameter, threshold,):

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
			for ipb, pred_pos in enumerate(pred):
				for igb, gt_pos in enumerate(gt):
					iou = calc_iou_individual(pred_pos, gt_pos, diameter)
					if iou > threshold:
						gt_idx_thresh.append(igb)
						pred_idx_thresh.append(ipb)
						ious.append(iou)

			# sort by iou
			args_desc = np.argsort(ious)[::-1]

			if len(args_desc) == 0:
				# No matches
				tp = 0
				fp = len(pred)
				fn = len(gt)
			else:
				gt_match_idx = []
				pred_match_idx = []
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

		def get_precision_recall(ground_truths, predictions, diameter, threshold):

			precisions = []
			recalls = []
			for gt, pred in zip(ground_truths, predictions):
				tp, fp, fn = get_results(gt, pred, diameter, threshold,)

				try:
					prec = tp/(tp + fp)
				except ZeroDivisionError:
					prec = 0.0
				try:
					rec = tp/(tp + fn)
				except ZeroDivisionError:
					rec = 0.0
				
				precisions.append(prec)
				recalls.append(rec)
				
			return np.mean(precisions), np.mean(recalls)

		print('Calculating average precision, threshold:')

		precisions = []
		recalls = []
		thresholds = np.linspace(0,1,11)
		for thresh in thresholds:
			print(thresh)
			prec, rec = get_precision_recall(ground_truths, predictions, diameter, thresh,)
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
		avg_prec = np.mean(prec_at_rec)

		return precisions, recalls, avg_prec




