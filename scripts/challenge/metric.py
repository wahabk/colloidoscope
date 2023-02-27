import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
from typing import Union
# this lazy imports numba in case you get issues with it

def _calc_iou_dist(distances, diameter, threshold):
	"""
	Measure distance between two particles normalised by diameters
	"""

	gt_idx_thresh = []
	pred_idx_thresh = []
	ious = []
	for ipb, pred_list in enumerate(distances):
		for igb, dist in enumerate(pred_list):

			iou_dist = 1 - (dist/diameter)
			if iou_dist < 0 : iou_dist = 0
			iou = iou_dist
			if iou > threshold:
				gt_idx_thresh.append(igb)
				pred_idx_thresh.append(ipb)
				ious.append(iou)
	return gt_idx_thresh, pred_idx_thresh, ious

def _get_results(gt, pred, diameter, threshold,) -> tuple:
	"""
	Find true positives, false positives, and false negatives from ground truth and predictions
	"""

	from numba import njit # lazy import numba

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
	jitted_calc_iou_dist = njit(_calc_iou_dist)
	gt_idx_thresh, pred_idx_thresh, ious = jitted_calc_iou_dist(dists, diameter, threshold)
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
			# If the boxes have not been matched already, add them to matches
			if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
				gt_match_idx.append(gt_idx)
				pred_match_idx.append(pr_idx)
				iou_index.append(i)
				iou_at_pred.append(ious[idx])
		
		tp = len(gt_match_idx) # ones that are in both
		fp = len(pred) - len(pred_match_idx) # predictions not matched
		fn = len(gt) - len(gt_match_idx) # grount truths not in pred

	return tp, fp, fn

def get_precision_recall(ground_truths:np.ndarray, predictions:np.ndarray, diameter:int, threshold:float, canvas_size:tuple) -> tuple:
	"""Take true positions and predictions and find precision and recall from TP, FP, FN

	Works only in 3D

	Args:
		ground_truths (np.ndarray): True positions.
		predictions (np.ndarray): Predicted positions.
		diameter (int): Diameter of the particles.
		threshold (float): The threshold is the fraction of the diameter to consider a detection to be true or false.
		exclude_borders (bool, optional): Whether to remove the predicitions around the image border. Defaults to True.
		canvas_size (tuple, optional): This is required to remove the predictions around the borders. Defaults to (64,64,64).

	Raises:
		ValueError: If there are NaN values in the ground truth or predictions

	Returns:
		tuple: a tuple of the precision and recall in this order (P, R)
	"""
	#
	
	
	if np.isnan(ground_truths).any(): raise ValueError("You have provided a ground truth array with NaN values, please remove these beforehands")
	if np.isnan(predictions).any(): raise ValueError("You have provided a ground truth array with NaN values, please remove these beforehands")

	ground_truths = remove_borders(ground_truths, canvas_size, pad=diameter/2,)
	predictions = remove_borders(predictions, canvas_size, pad=diameter/2,)
	
	tp, fp, fn = _get_results(ground_truths, predictions, diameter, threshold)

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

def average_precision(ground_truth:np.ndarray, prediction:np.ndarray, diameter:int, canvas_size:tuple) -> tuple:
	"""Measure average precision in 3D for spheres

	This will sweep through 10 thresholds of 0 to 1 in 0.1 increments
	
	Based on https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734

	Args:
		ground_truth (np.ndarray): True positions.
		prediction (np.ndarray): Predicted positions.
		diameter (int): Diameter of the particles.
		canvas_size (tuple, optional): This is required to remove the predictions around the borders. Defaults to (64,64,64).

	Raises:
		ValueError: If there are NaN values in the ground truth or predictions.

	Returns:
		tuple: AP, precisions, recalls

		The first variable is the scalar AP value (between 0 and 1), then the precisions and recalls are provided if you want to plot a PR curve
	"""

	print('Calculating average precision, threshold:')
	precisions = []
	recalls = []
	thresholds = np.linspace(0,1,10)
	for thresh in tqdm(thresholds):
		prec, rec = get_precision_recall(ground_truth, prediction, diameter, thresh, canvas_size)
		precisions.append(prec)
		recalls.append(rec)
	print('Done')

	precisions = np.array(precisions)
	recalls = np.array(recalls)
	recalls = np.flip(recalls)

	ap = np.trapz(precisions, x=recalls) # integrate

	return ap, precisions, recalls

def plot_pr(ap, precisions, recalls, name, tag='o-', *args, **kwargs):

	plt.plot(recalls, precisions, tag, label=f'{name} AP = {ap:.2f}', *args, **kwargs)
	plt.title('Average Precision')
	plt.xlim([-0.1,1.1])
	plt.ylim([-0.1,1.1])
	plt.legend()
	return plt.gcf()

def remove_borders(centers:np.ndarray, canvas_size:tuple, pad:Union[int, float], diameters:np.ndarray=None, label_size:tuple=None):
	
	if label_size is not None:
		zdiff = (canvas_size[0] - label_size[0])/2
		xdiff = (canvas_size[1] - label_size[1])/2
		ydiff = (canvas_size[2] - label_size[2])/2
		# print(centers[0])
		centers = centers - [zdiff, xdiff, ydiff]

	indices = []
	# pad = 0
	for idx, c in enumerate(centers):
		if pad<=c[0]<=(canvas_size[0]-pad) and pad<=c[1]<=(canvas_size[1]-pad) and pad<=c[2]<=(canvas_size[2]-pad):
			indices.append(idx)

	final_centers = centers[indices]

	if diameters is not None:
		final_diameters = diameters[indices]
		return final_centers, final_diameters
	else:
		return final_centers
