import numpy as np
import matplotlib.pyplot as plt 
import trackpy as tp
from sklearn import metrics
from colloidoscope.trainer import predict
from colloidoscope.unet import UNet
from colloidoscope.deepcolloid import DeepColloid
import torch
from scipy.spatial.distance import pdist

def run_trackpy(array, diameter=11):
	f = tp.locate(array, diameter)
	f = [list(f[:]['z']), list(f[:]['x']), list(f[:]['y'])]
	new = []
	for i in range(0, len(f[0])):
		new.append([ f[0][i], f[2][i], f[1][i] ])
	tp_predictions = np.array(new)

	return tp_predictions

def gr(positions, cutoff, bins, minimum_gas_number=1):
	# from yushi yang



	bins = np.linspace(0, cutoff, bins)
	drs = bins[1:] - bins[:-1]
	distances = pdist(positions).ravel()

	if positions.shape[0] < minimum_gas_number:
		rg_hists = []
		for i in range(int(minimum_gas_number) // positions.shape[0] + 2):
			random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
			rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]
			rg_hists.append(rg_hist)
		rg_hist = np.mean(rg_hists, 0)

	else:
		random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
		rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]

	hist = np.histogram(distances, bins=bins)[0]

	print(hist, rg_hist)

	rg_hist[rg_hist==0] = 0.001

	hist = hist / rg_hist # pdfs
	hist[np.isnan(hist)] = 0
	bin_centres = (bins[1:] + bins[:-1]) / 2
	return bin_centres, hist # as x, y

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

	canvas, metadata, gt_positions = dc.read_hdf5(dataset_name, 3002, read_metadata=True)
	print(metadata)
	diameters = [metadata['params']['r']*2 for i in range(len(gt_positions))]

	tp_predictions = run_trackpy(canvas, diameter=metadata['params']['r']*2-1)
	print(metadata)
	print(gt_positions.shape, tp_predictions.shape)

	# dc.view(canvas, tp_predictions)

	x, y = gr(gt_positions, 7, 25)
	plt.plot(x, y, label='true')
	x, y = gr(tp_predictions, 7, 25)
	plt.plot(x, y, label='pred')
	plt.legend()
	plt.show()

	# precisions, recalls, thresholds, predictions = dc.average_precision(gt_positions, tp_predictions, diameters=diameters)
	# print(precisions, recalls)

	# # display = metrics.PrecisionRecallDisplay.from_predictions(true, predictions) #(precision=precisions, recall=recalls, estimator_name='unet').plot()
	# display = metrics.PrecisionRecallDisplay(precision=np.flip(precisions), recall=recalls, estimator_name='trackpy').plot()
	# plt.xlim([-0.1,1.1])
	# plt.ylim([-0.1,1.1])
	# # plot_precision_recall_curve(true, predictions)
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



