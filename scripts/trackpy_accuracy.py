import numpy as np
import matplotlib.pyplot as plt 
import trackpy as tp
from sklearn import metrics
from colloidoscope.trainer import predict
from colloidoscope.unet import UNet
from colloidoscope.deepcolloid import DeepColloid
import torch

def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    ax.plot(recalls, precisions, 'rx')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])
    return ax

def plot_precision_recall_curve(self):
	#http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
	precision, tpr_recall, _ = metrics.precision_recall_curve(y_true, y_score)
	pr_curve, = plt.step(tpr_recall, precision, color='r', alpha=0.2, where='post',linewidth=lw,linestyle=main_linestyle)
	plt.fill_between(tpr_recall, precision, step='post', alpha=0.2, color='r')
	plt.set_xlabel('True Positive Rate (Recall)')
	plt.set_ylabel('Precision')
	plt.set_ylim([0.0, 1.05])
	plt.set_xlim([0.0, 1.0])
	plt.set_title('Average Precision=%0.2f' % metrics.average_precision_score(y_true, y_score))
	
	#Plot dots at certain decision thresholds, for clarity
	for d in [0.1, 0.5, 0.9]:
		tpr_recall, _, precision = calculate_tpr_fpr_prec(y_true, y_score, d)
		plt.plot(tpr_recall, precision, 'o', color = pr_color)
		text = plt.annotate('d='+str(d), (tpr_recall, precision))
		text.set_rotation(45)

def run_trackpy(array, diameter=11):
	f = tp.locate(array, diameter)
	f = [list(f[:]['z']), list(f[:]['x']), list(f[:]['y'])]
	new = []
	for i in range(0, len(f[0])):
		new.append([ f[0][i], f[2][i], f[1][i] ])
	tp_predictions = np.array(new)

	return tp_predictions

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

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)
	dataset_name = 'new_year'

	canvas, metadata, gt_positions = dc.read_hdf5(dataset_name, 3007, read_metadata=True)
	print(metadata)
	diameters = [metadata['params']['r']*2 for i in range(len(gt_positions))]

	tp_predictions = run_trackpy(canvas, diameter=metadata['params']['r']*2+1)
	print(metadata)
	print(gt_positions.shape, tp_predictions.shape)

	precisions, recalls, thresholds, predictions = dc.average_precision(gt_positions, tp_predictions, diameters=diameters)
	true = [1 for i in gt_positions]
	print(precisions, recalls)
	print(thresholds)

	# display = metrics.PrecisionRecallDisplay.from_predictions(true, predictions) #(precision=precisions, recall=recalls, estimator_name='unet').plot()
	display = metrics.PrecisionRecallDisplay(precision=precisions, recall=np.flip(recalls), estimator_name='trackpy').plot()
	plt.xlim([-0.1,1.1])
	plt.ylim([-0.1,1.1])
	plt.show()

	# dc.view(canvas, tp_predictions)

	# display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, estimator_name='Trackpy').plot()
	# fig = plt.figure()
	# ax = plt.gca()
	# ax = plot_pr_curve(precisions, recalls, category='trackpy')
	# plt.savefig('output/roc_trackpy.png')

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# print(f'predict on {device}')
	# # model
	# model = UNet(in_channels=1,
	# 			out_channels=1,
	# 			n_blocks=4,
	# 			start_filters=32,
	# 			activation='relu',
	# 			normalization='batch',
	# 			conv_mode='same',
	# 			dim=3).to(device)

	# roiSize = (32,128,128)
	# threshold = 0.5
	# weights_path =  'output/weights/unet.pt'
	

	# canvas, gt_positions = dc.read_hdf5(dataset_name, 202, return_positions=True)
	# label, tp_predictions = predict(canvas, model, device, weights_path, threshold=threshold, return_positions=True)
	# print(gt_positions.shape, tp_predictions.shape)
	# ap, precisions, recalls = dc.average_precision(gt_positions, tp_predictions, diameter=9)
	# recalls = np.array([1-r for r in recalls])
	# print(ap, precisions, recalls)

	# display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, estimator_name='unet').plot()
	# fig = display.figure_

	# precisions = np.array([1-p for p in precisions])
	# fig = plt.figure()
	# ax = plt.gca()
	# thresholds = np.linspace(0.05,0.95,10)
	# ax = plot_pr_curve(precisions, recalls, category='unet')
	# plt.savefig('output/roc_unet.png')

	# print(aps)
	# print(np.mean(aps))
	# plt.show()
	# fig = plot_pr_curve(precisions, recalls)



