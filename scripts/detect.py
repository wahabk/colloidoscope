import colloidoscope as cd
import napari
import matplotlib.pyplot as plt

if __name__ == '__main__':
	dataset_path = dataset_path = '/home/wahab/Data/HDD/Colloids/'

	dataset_name = 'newyear'

	dc = cd.DeepColloid(dataset_path)
	
	# array = dc.crop3d(array, (32,128,128))

	# reader = explore_lif.Reader('Data/i-ii.lif')
	# series = reader.getSeries()
	# chosen = series[5]  # choose first image in the lif file
	# # get a numpy array corresponding to the 1st time point & 1st channel
	# # the shape is (x, y, z)
	# video = [chosen.getXYZ(T=t, channel=0) for t in range(chosen.getNbFrames())]
	# video = np.array(video)
	# print(video.shape)
	# t, x, y, z


	# data = dc.read_hdf5(dataset_name, 645)
	# true_positions = data['positions']
	# array = data['image']
	# diameters = data['diameters']
	# print(data['metadata'])

	# pos, label = dc.detect(array, debug=True)
	# print(pos.shape)
	# dc.view(array, positions=pos, label=label)


	# x, y = dc.get_gr(true_positions, 7, 25, minimum_gas_number=1)
	# plt.plot(x, y, label='true')
	# x, y = dc.get_gr(pos, 7, 25, minimum_gas_number=1)
	# plt.plot(x, y, label='pred')
	# plt.legend()
	# plt.show()
	# plt.clf()

	# ap, precisions, recalls, thresholds = dc.average_precision(true_positions, pos, diameters=diameters)

	# fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='Unet')
	# plt.show()
	# plt.clf()

	path = '/home/wahab/Data/HDD/Colloids/Real/Levke/LS51_4_binary_30nm_smallRange_decon.tif'	
	array = dc.read_tif(path)

	import pdb; pdb.set_trace()

	pos, label = dc.detect(array, debug=True)
	print(pos.shape)
	dc.view(array, positions=pos, label=label)

	x, y = dc.get_gr(pos, 50, 30)
	plt.plot(x, y, label='pred')
	plt.show()

	# viewer=napari.Viewer()
	# viewer.add_image(array)
	# napari.run()	

	# import pdb;pdb.set_trace()