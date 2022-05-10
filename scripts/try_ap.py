import colloidoscope as cd
import napari
import matplotlib.pyplot as plt
import numpy as np
import scipy
import trackpy as tp
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist
from numba import njit

if __name__ == '__main__':
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	dataset_path = '/home/ak18001/Data/HDD/Colloids'

	dataset_name = 'test'

	dc = cd.DeepColloid(dataset_path)

	data_dict = dc.read_hdf5(dataset_name, 1)
	array, true_positions, label, diameters, metadata = data_dict['image'],	data_dict['positions'],	data_dict['label'],	data_dict['diameters'],	data_dict['metadata']
	print(metadata)
	print(dc.round_up_to_odd(metadata['params']['r']*2))

	# dc.view(array, positions=true_positions, label=label)

	trackpy_pos, df = dc.run_trackpy(array, diameter = 5)
	trackpy_on_label, df = dc.run_trackpy(label, diameter = 5)

	print(true_positions.shape)
	print(trackpy_pos.shape)
	print(trackpy_on_label.shape)

	x, y = dc.get_gr(true_positions, 50, 100)
	plt.plot(x, y, label=f'true (n={len(true_positions)})')
	x, y = dc.get_gr(trackpy_pos, 50, 100)
	plt.plot(x, y, label=f'tp sim (n={len(trackpy_pos)}')
	x, y = dc.get_gr(trackpy_on_label, 50, 100)
	plt.plot(x, y, label=f'tp label (n={len(trackpy_on_label)}')
	plt.legend()
	# plt.show()
	plt.savefig('output/test/gr.png')
	plt.clf()

	# dc.view(array, label=label, positions=true_positions)
	# dc.view(array, label=label, positions=trackpy_pos)
	# dc.view(array, label=label, positions=trackpy_on_label)

	tp_ap, precisions, recalls, thresholds = dc.average_precision(true_positions, trackpy_pos, diameters)
	fig = dc.plot_pr(tp_ap, precisions, recalls, thresholds, name='tp sim', tag='bo-')
	label_ap, precisions, recalls, thresholds = dc.average_precision(true_positions, trackpy_on_label, diameters)
	fig = dc.plot_pr(label_ap, precisions, recalls, thresholds, name='tp label', tag='ro-')
	# plt.show()
	plt.savefig('output/test/ap.png')