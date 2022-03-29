import colloidoscope as cd
import napari
import matplotlib.pyplot as plt
import numpy as np
import scipy
import trackpy as tp
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist
from numba import njit


@njit
def calc_iou_dist(distances, diameters, threshold):
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

def get_results(gt, pred, diameters, threshold,):

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
	gt_idx_thresh, pred_idx_thresh, ious = calc_iou_dist(dists, diameters, threshold)
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
	tp, fp, fn = get_results(ground_truths, predictions, diameters, threshold)

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
		print(thresh)
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

def run_trackpy(array, diameter=5, *args, **kwargs):
	df = None
	df = tp.locate(array, diameter=5, *args, **kwargs)
	f = list(zip(df['z'], df['y'], df['x']))
	tp_predictions = np.array(f)

	return tp_predictions

if __name__ == '__main__':
	dataset_path = '/home/wahab/Data/HDD/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'

	dataset_name = 'test'

	dc = cd.DeepColloid(dataset_path)

	data_dict = dc.read_hdf5(dataset_name, 1)
	array, true_positions, label, diameters, metadata = data_dict['image'],	data_dict['positions'],	data_dict['label'],	data_dict['diameters'],	data_dict['metadata']
	print(metadata)
	print(dc.round_up_to_odd(metadata['params']['r']*2))

	trackpy_pos = run_trackpy(array/array.max(), diameter = dc.round_up_to_odd(metadata['params']['r']*2))
	trackpy_on_label = run_trackpy(label, diameter = dc.round_up_to_odd(metadata['params']['r']*2))

	print(array.shape, label.shape)
	print(array.max(), label.max())

	print(trackpy_pos.shape)
	print(trackpy_on_label.shape)
	print(trackpy_pos[0])
	print(trackpy_on_label[0])

	# x, y = dc.get_gr(true_positions, 50, 100)
	# plt.plot(x, y, label=f'true (n={len(true_positions)})')
	# x, y = dc.get_gr(trackpy_positions, 50, 100)
	# plt.plot(x, y, label=f'tp sim (n={len(trackpy_positions)}')
	# x, y = dc.get_gr(trackpy_on_label, 50, 100)
	# plt.plot(x, y, label=f'tp label (n={len(trackpy_on_label)}')
	# plt.legend()
	# plt.show()
	# plt.clf()

	tp_ap, precisions, recalls, thresholds = average_precision(true_positions, trackpy_pos, diameters)
	print(precisions, recalls)
	fig = dc.plot_pr(tp_ap, precisions, recalls, thresholds, name='tp sim')
	plt.show()
	print(diameters.shape, diameters[0])
	label_ap, precisions, recalls, thresholds = average_precision(true_positions, trackpy_on_label, diameters)
	print(diameters.shape, diameters[0])
	fig = dc.plot_pr(label_ap, precisions, recalls, thresholds, name='tp label')
	plt.show()