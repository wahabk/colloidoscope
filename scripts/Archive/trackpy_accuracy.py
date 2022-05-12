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
		new.append([ f[0][i], f[1][i], f[2][i] ])
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

	# print(hist, rg_hist)

	rg_hist[rg_hist==0] = 0.001

	hist = hist / rg_hist # pdfs
	hist[np.isnan(hist)] = 0
	bin_centres = (bins[1:] + bins[:-1]) / 2
	return bin_centres, hist # as x, y





if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)
	dataset_name = 'janpoly'

	# canvas, label, gt_positions, metadata = dc.read_hdf5_old(dataset_name, 3007)
	data = dc.read_hdf5(dataset_name, 171)
	canvas, metadata, gt_positions = data['image'], data['metadata'], data['positions']
	print(metadata)
	diameters = [metadata['params']['r']*2 for i in range(len(gt_positions))]

	tp_predictions = run_trackpy(canvas, diameter=metadata['params']['r']*2-1)


	df, unet_pos, label = dc.detect(canvas, debug=True)
	print(gt_positions.shape, tp_predictions.shape, unet_pos.shape)
	print(gt_positions[0], tp_predictions[0], unet_pos[0])

	# dc.view(canvas, tp_predictions)

	x, y = dc.get_gr(gt_positions, 50, 50)
	plt.plot(x, y, label='true')
	x, y = dc.get_gr(tp_predictions, 50, 50)
	plt.plot(x, y, label='trackpy')
	x, y = dc.get_gr(unet_pos, 50, 50)
	plt.plot(x, y, label='unet')
	plt.legend()
	# plt.show()
	plt.savefig('output/gr.png')

	plt.clf()

	ap, precisions, recalls, thresholds = dc.average_precision(gt_positions, tp_predictions, diameters=diameters)
	fig = dc.plot_pr(ap, precisions=precisions, recalls=recalls, thresholds=thresholds, name='Trackpy', tag='bo-')
	ap, precisions, recalls, thresholds = dc.average_precision(gt_positions, unet_pos, diameters=diameters)
	fig = dc.plot_pr(ap, precisions=precisions, recalls=recalls, thresholds=thresholds, name='Unet', tag='ro-')
	# plt.show()
	plt.savefig('output/roc_trackpy.png')
	

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
	



