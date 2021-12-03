import numpy as np
import matplotlib.pyplot as plt 
import trackpy as tp
from src.deepcolloid import DeepColloid
from sklearn import metrics

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
	for i in range(20, 45):
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

	precisions, recalls, ap = dc.average_precision(multi_gt_pos, multi_pred_pos)
	
	# recalls = np.array([1-r for r in recalls])
	# precisions = np.array([1-p for p in precisions])

	display = metrics.PrecisionRecallDisplay(precision=precisions, recall=recalls, average_precision=ap, estimator_name='Trackpy').plot()
	fig = display.figure_

	plt.savefig('output/trackpyroc.png')

	# plt.show()
	# fig = plot_pr_curve(precisions, recalls)



