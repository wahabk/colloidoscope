import colloidoscope as cd
import napari
import matplotlib.pyplot as plt
import numpy as np
import scipy
import trackpy as tp
import monai
import math

def run_trackpy(array, diameter=5, *args, **kwargs):
	df = tp.locate(array, diameter=5, *args, **kwargs)
	f = list(zip(df['z'], df['y'], df['x']))
	tp_predictions = np.array(f)

	return tp_predictions

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1 # // is floor division

if __name__ == '__main__':
	dataset_path = '/home/wahab/Data/HDD/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'

	dataset_name = 'test'

	dc = cd.DeepColloid(dataset_path)

	# data_dict = dc.read_hdf5(dataset_name, 1)
	# array, true_positions, label, diameters, metadata = data_dict['image'],	data_dict['positions'],	data_dict['label'],	data_dict['diameters'],	data_dict['metadata']
	# print(metadata)
	# print(round_up_to_odd(metadata['params']['r']*2))

	# trackpy_positions = run_trackpy(array, diameter = round_up_to_odd(metadata['params']['r']*2))
	# trackpy_on_label = run_trackpy(label, diameter = round_up_to_odd(metadata['params']['r']*2))


	# x, y = dc.get_gr(true_positions, 50, 100)
	# plt.plot(x, y, label=f'true (n={len(true_positions)})')
	# x, y = dc.get_gr(trackpy_positions, 50, 100)
	# plt.plot(x, y, label=f'tp sim (n={len(trackpy_positions)}')
	# x, y = dc.get_gr(trackpy_on_label, 50, 100)
	# plt.plot(x, y, label=f'tp label (n={len(trackpy_on_label)}')
	# plt.legend()
	# plt.show()

	# tp_ap, precisions, recalls, thresholds = dc.average_precision(true_positions, trackpy_positions, diameters)
	# fig = dc.plot_pr(tp_ap, precisions, recalls, thresholds, name='tp sim')
	# plt.show()
	# label_ap, precisions, recalls, thresholds = dc.average_precision(true_positions, trackpy_positions, diameters)
	# fig = dc.plot_pr(label_ap, precisions, recalls, thresholds, name='tp label')
	# plt.show()



	# path = '/home/wahab/Data/HDD/Colloids/Real/Levke/Levke_smallParticles_betterData_2021_4_1/goodData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	path = '/home/ak18001/Data/HDD/Colloids/Real/Levke/goodData_2021_4_1Levke_smallParticles_betterData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	array = dc.read_tif(path)
	print(array.shape)
	# array = array[:,0,:,:]
	array = array[0]
	# array = dc.crop3d(array, (32,128,128))

	start_filters = 32
	n_blocks = 3

	start = int(math.sqrt(start_filters))
	channels = [2**n for n in range(start, start + n_blocks)]
	strides = [2 for n in range(1, n_blocks)]

	model = monai.networks.nets.AttentionUnet(
	spatial_dims=3,
	in_channels=1,
	out_channels=1,
	channels=channels,
	strides=strides,
	dropout=0
	)

	df, pos, result = dc.detect(array, diameter=9, patch_overlap=(16,16,16), threshold=0.5, debug=True, model=model, weights_path='output/weights/attention_unet_20220524.pt')
	print(pos.shape)
	dc.view(array, positions=pos, label=result)

	x, y = dc.get_gr(pos, 100, 100)
	plt.plot(x, y, label='pred')
	plt.show()

	# dc.view(array, pos, result)
	# import pdb;pdb.set_trace()