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

def run_trackpy(array, diameter=11):
	f = tp.locate(array, diameter)
	f = [list(f[:]['z']), list(f[:]['y']), list(f[:]['x'])]
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

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predict on {device}')

	# model
	model = UNet(in_channels=1,
				out_channels=1,
				n_blocks=4,
				start_filters=32,
				activation='relu',
				normalization='batch',
				conv_mode='same',
				dim=3).to(device)

	roiSize = (32,128,128)
	threshold = 0.5
	weights_path =  'output/weights/unet.pt'
	dataset_name = 'third_run'  


	canvas, gt_positions = dc.read_hdf5(dataset_name, 202, return_positions=True)
	tp_predictions = run_trackpy(canvas, diameter=9)
	print(gt_positions.shape, tp_predictions.shape)
	ap, precisions, recalls = dc.average_precision(gt_positions, tp_predictions, diameter=9)
	# recalls = np.array([1-r for r in recalls])
	print(ap, precisions, recalls)

	display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, estimator_name='Trackpy').plot()
	# fig = plt.figure()
	# ax = plt.gca()
	# ax = plot_pr_curve(precisions, recalls, category='trackpy')
	plt.savefig('output/roc_trackpy.png')

	canvas, gt_positions = dc.read_hdf5(dataset_name, 202, return_positions=True)
	label, tp_predictions = predict(canvas, model, device, weights_path, threshold=threshold, return_positions=True)
	print(gt_positions.shape, tp_predictions.shape)
	ap, precisions, recalls = dc.average_precision(gt_positions, tp_predictions, diameter=9)
	# recalls = np.array([1-r for r in recalls])
	print(ap, precisions, recalls)

	display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, estimator_name='unet').plot()
	fig = display.figure_

	# precisions = np.array([1-p for p in precisions])
	# fig = plt.figure()
	# ax = plt.gca()
	# thresholds = np.linspace(0.05,0.95,10)
	# ax = plot_pr_curve(precisions, recalls, category='unet')
	plt.savefig('output/roc_unet.png')

	# print(aps)
	# print(np.mean(aps))
	# plt.show()
	# fig = plot_pr_curve(precisions, recalls)



