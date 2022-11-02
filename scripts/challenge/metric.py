from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
# from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import numpy as np
import random
import psf
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path
from numba import njit

from scipy.spatial.distance import pdist, cdist

from tqdm import tqdm


@njit
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



def _get_results(gt, pred, diameters, threshold,):
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
	gt_idx_thresh, pred_idx_thresh, ious = _calc_iou_dist(dists, diameters, threshold)
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

def get_precision_recall(ground_truths, predictions, diameters, threshold):

	# take block of different images and find result
	tp, fp, fn = _get_results(ground_truths, predictions, diameters, threshold)

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

def average_precision(ground_truth, prediction, diameters):

	# based on https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734


	print('Calculating average precision, threshold:')

	precisions = []
	recalls = []
	predictions = []
	thresholds = np.linspace(0,1,10)
	for thresh in tqdm(thresholds):
		prec, rec = get_precision_recall(ground_truth, prediction, diameters, thresh,)
		precisions.append(prec)
		recalls.append(rec)
		# predictions.append(pred) # predictions are here for testing sklearn.metrics from_estimator
	print('Done')

	precisions = np.array(precisions)
	recalls = np.array(recalls)
	recalls = np.flip(recalls) # TODO why do i need to flip ????

	ap = np.trapz(precisions, x=recalls) # integrate

	return ap, precisions, recalls, thresholds




def crop_positions_for_label(centers, canvas_size, label_size, diameters):

	zdiff = (canvas_size[0] - label_size[0])/2
	xdiff = (canvas_size[1] - label_size[1])/2
	ydiff = (canvas_size[2] - label_size[2])/2
	print(centers[0])
	# centers = centers - [zdiff, xdiff, ydiff]
	print(centers[0])
	print([zdiff, xdiff, ydiff])
	
	indices = []
	pad = 0
	for idx, c in enumerate(centers):
		if pad<=c[0]<=(label_size[0]-pad) and pad<=c[1]<=(label_size[1]-pad) and pad<=c[2]<=(label_size[2]-pad):
			indices.append(idx)

	print(centers[0])
	centers = centers[indices]
	diameters = diameters[indices]

	return centers, diameters

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	canvas_size=(64,64,64)
	label_size=(60,60,60)


	r = 5
	d = r*2
	index = 1
	volfrac = 0.1

	path = f"{dataset_path}/Positions/phi{volfrac*1000:.0f}.gsd"
	hoomd_positions, diameters = read_gsd(path, index)

	pos, diameters = convert_hoomd_positions(hoomd_positions, canvas_size=canvas_size, diameters=diameters, diameter=d)

	new_pos, _ = crop_positions_for_label(pos, canvas_size=canvas_size, label_size=label_size, diameters=diameters)
	print(pos.shape, new_pos.shape)
	# print(pos[0], new_pos[0])
	# print(diameters.min())
	# prec, rec = get_precision_recall(pos, pos[:int(len(pos)/4)], diameters=diameters, threshold=0.5)
	prec, rec = get_precision_recall(pos, new_pos, diameters=diameters, threshold=0.5)

	print(prec, rec)