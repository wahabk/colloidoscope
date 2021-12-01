import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage, spatial
from torch._C import Value
import trackpy as tp
from src.deepcolloid import DeepColloid
import math


def calc_iou_individual(center1, center2, diameter=10,):
	r = diameter / 2
	c = math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2 + (center1[2]-center2[2])**2)
	area = math.pi*r**2

 	q = 2* math.acos(math.radians(c/2*r))
	overlap = (r**2)*(q - math.sin(q)) # area of overlap of circles with same r

	print(c, r, overlap)
	iou = overlap / (area + area - overlap)
	return iou

def get_results(gt, pred, threshold, diameter,):

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
			print(pred_pos, gt_pos)
			iou = calc_iou_individual(pred_pos, gt_pos, diameter)
			if iou > threshold:
				gt_idx_thresh.append(igb)
				pred_idx_thresh.append(ipb)
				ious.append(iou)

	args_desc = np.argsort(ious)[::-1]
	print(args_desc)
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
			# If the boxes are unmatched, add them to matches
			if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
				gt_match_idx.append(gt_idx)
				pred_match_idx.append(pr_idx)
		tp = len(gt_match_idx)
		fp = len(pred) - len(pred_match_idx)
		fn = len(gt) - len(gt_match_idx)

	return tp, fp, fn

def get_precision_recall(gt, pred, diameter=10, threshold=0.5,):
	
	tp, fp, fn = get_results(gt, pred, threshold, diameter)

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

def mAP(ground_truths, predictions, diameter=10, threshold=0.5,):

	precisions = []
	recalls = []
	for gt, pred in zip(ground_truths, predictions):
		p, r = get_precision_recall(gt, pred, diameter, threshold)
		precisions.append(p)
		recalls.append(r)
	precisions =  np.array(precisions)
	recalls =  np.array(recalls)

	prec_at_rec = []
	for recall_level in np.linspace(0.0, 1.0, 11):
		try:
			args = np.argwhere(recalls >= recall_level).flatten()
			prec = max(precisions[args])
		except ValueError:
			prec = 0.0
		prec_at_rec.append(prec)
	avg_prec = np.mean(prec_at_rec)

	# precision recall

	return precisions, recalls, avg_prec

def plot_pr_curve(precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    ax.scatter(recalls, precisions, label=label, s=20)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall')
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax



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
	threshold = 0.3
	weights_path =  'output/weights/unet.pt'
	dataset_name = 'replicate'


	multi_gt_pos = []
	multi_pred_pos = []
	for i in range(40, 46):
		canvas, gt_positions = dc.read_hdf5(dataset_name, i, return_positions=True)

		f = tp.locate(canvas, 11)
		f = [list(f[:]['z']), list(f[:]['y']), list(f[:]['x'])]

		new = []
		for i in range(0, len(f[0])):
			new.append([ f[0][i], f[1][i], f[2][i] ])
		tp_predictions = np.array(new)

		multi_gt_pos.append(gt_positions)
		multi_pred_pos.append(tp_predictions)

	precisions, recalls, avg_prec = mAP(multi_gt_pos, multi_pred_pos)

	ax = plot_pr_curve(precisions, recalls)
	plt.show()



