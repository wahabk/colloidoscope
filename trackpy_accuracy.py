import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from torch._C import Value
import trackpy as tp
from src.deepcolloid import DeepColloid
import math
from sklearn import metrics


def calc_iou_individual(center1, center2, diameter,):
	r = diameter / 2
	c = math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2 + (center1[2]-center2[2])**2)
	area = math.pi*r**2
	try:
		q = 2* math.acos(math.radians(c/2*r))
		overlap = (r**2)*(q - math.sin(q)) # area of overlap of circles with same r
	except ValueError:
		overlap = 0
	iou = overlap / (area + area - overlap)
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

def average_precision(ground_truths, predictions, diameter=10):

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

	return precisions, recalls

def plot_pr_curve(precisions, recalls, label=None):
	"""Simple plotting helper function"""


	fig, ax = plt.subplots(figsize=(6,6))
	ax.plot(recalls, precisions, label='Trackpy')
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.legend(loc='center left')
	ax.set_xlim(0,1.1)
	ax.set_ylim(0,1.1)

	return fig



if __name__ == '__main__':
	# reader = explore_lif.Reader('Data/i-ii.lif')
	# series = reader.getSeries()
	# chosen = series[5]  # choose first image in the lif file
	# # get a numpy array corresponding to the 1st time point & 1st channel
	# # the shape is (x, y, z)
	# video = [chosen.getXYZ(T=t, channel=0) for t in range(chosen.getNbFrames())]
	# video = np.array(video)
	# print(video.shape)
	# t, x, y, z

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	roiSize = (32,128,128)
	threshold = 0.9
	weights_path =  'output/weights/unet.pt'
	dataset_name = 'replicate'


	multi_gt_pos = []
	multi_pred_pos = []
	for i in range(40, 45):
		canvas, gt_positions = dc.read_hdf5(dataset_name, i, return_positions=True)

		f = tp.locate(canvas, 9)
		f = [list(f[:]['z']), list(f[:]['y']), list(f[:]['x'])]
		new = []
		for i in range(0, len(f[0])):
			new.append([ f[0][i], f[1][i], f[2][i] ])
		tp_predictions = np.array(new)

		multi_gt_pos.append(gt_positions)
		multi_pred_pos.append(tp_predictions)
	multi_gt_pos = np.array(multi_gt_pos)
	multi_pred_pos = np.array(multi_pred_pos)

	print(multi_gt_pos[0].shape, multi_pred_pos[0].shape)

	precisions, recalls = average_precision(multi_gt_pos, multi_pred_pos)
	
	# recalls = np.array([1-r for r in recalls])

	print(precisions, recalls)
	# roc_auc = metrics.auc(precisions, recalls)
	# display = metrics.RocCurveDisplay.from_predictions(fpr=recalls, tpr=precisions, roc_auc=roc_auc,
	# 		estimator_name='Trackpy')
	# display.plot()
	display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, estimator_name='Trackpy').plot()

	plt.savefig('output/trackpyroc.png')

	# plt.show()
	# fig = plot_pr_curve(precisions, recalls)



