import numpy as np
import matplotlib.pyplot as plt 
import trackpy as tp
from sklearn import metrics
from colloidoscope import DeepColloid

def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    ax.scatter(recalls, precisions, label=label, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
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

	roiSize = (32,128,128)
	threshold = 0.5
	weights_path =  'output/weights/unet.pt'
	dataset_name = 'replicate'


	multi_gt_pos = []
	multi_pred_pos = []
	for i in range(20, 45):
		canvas, gt_positions = dc.read_hdf5(dataset_name, i, return_positions=True)

		tp_predictions = run_trackpy(canvas, diameter=11)

		multi_gt_pos.append(gt_positions)
		multi_pred_pos.append(tp_predictions)
	# multi_gt_pos = np.array(multi_gt_pos)
	# multi_pred_pos = np.array(multi_pred_pos)

	aps, precisions, recalls = dc.average_precision(multi_gt_pos, multi_pred_pos)

	print(precisions, recalls)
	# recalls = np.array([1-r for r in recalls])
	# precisions = np.array([1-p for p in precisions])

	# display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, average_precision=ap, estimator_name='Trackpy').plot()
	# fig = display.figure_

	fig = plt.figure()
	ax = plt.gca()
	thresholds = np.linspace(0.05,0.95,10)
	ax = plot_pr_curve(precisions, recalls, label=thresholds, category='Trackpy')
	plt.savefig('output/trackpyroc.png')

	print(aps)
	print(np.mean(aps))
	# plt.show()
	# fig = plot_pr_curve(precisions, recalls)



