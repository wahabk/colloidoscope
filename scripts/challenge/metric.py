import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
# this lazy imports numba in case you get issues with it

def _calc_iou_dist(distances, diameter, threshold):
	"""
	Measure distance between two spheres normalised by diameters
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
			# If the boxes have not been matched, add them to matches
			if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
				gt_match_idx.append(gt_idx)
				pred_match_idx.append(pr_idx)
				iou_index.append(i)
				iou_at_pred.append(ious[idx])
		
		tp = len(gt_match_idx) # ones that are in both
		fp = len(pred) - len(pred_match_idx) # predictions not matched
		fn = len(gt) - len(gt_match_idx) # grount truths not in pred


	return tp, fp, fn

def get_precision_recall(ground_truths, predictions, diameter, threshold):
	# take block of different predictions and find result
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

def average_precision(ground_truth, prediction, diameter):
	# based on https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734

	print('Calculating average precision, threshold:')

	precisions = []
	recalls = []
	predictions = []
	thresholds = np.linspace(0,1,10)
	for thresh in tqdm(thresholds):
		prec, rec = get_precision_recall(ground_truth, prediction, diameter, thresh,)
		precisions.append(prec)
		recalls.append(rec)
		# predictions.append(pred) # predictions are here for testing sklearn.metrics from_estimator
	print('Done')

	precisions = np.array(precisions)
	recalls = np.array(recalls)
	recalls = np.flip(recalls)

	ap = np.trapz(precisions, x=recalls) # integrate

	return ap, precisions, recalls, thresholds

def plot_pr(ap, precisions, recalls, thresholds, name, tag='o-', *args, **kwargs):
	# display = metrics.PrecisionRecallDisplay(precision=precisions, 
	# recall=recalls, estimator_name=name).plot()

	plt.plot(recalls, precisions, tag, label=f'{name} AP = {ap:.2f}', *args, **kwargs)
	plt.title('Average Precision')
	plt.xlim([-0.1,1.1])
	plt.ylim([-0.1,1.1])
	plt.legend()
	return plt.gcf()

def exclude_borders(centers, canvas_size, pad, diameters=None, label_size=None):
	
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
