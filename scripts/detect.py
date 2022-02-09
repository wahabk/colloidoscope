import colloidoscope as cd
import napari
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	# dataset_path = '/home/wahab/Data/HDD/Colloids/'
	dataset_path = '/home/ak18001/Data/HDD/Colloids'

	dataset_name = 'janpoly'

	dc = cd.DeepColloid(dataset_path)
	
	# reader = explore_lif.Reader('Data/i-ii.lif')
	# series = reader.getSeries()
	# chosen = series[5]  # choose first image in the lif file
	# # get a numpy array corresponding to the 1st time point & 1st channel
	# # the shape is (x, y, z)
	# video = [chosen.getXYZ(T=t, channel=0) for t in range(chosen.getNbFrames())]
	# video = np.array(video)
	# print(video.shape)
	# t, x, y, z

	# data = dc.read_hdf5(dataset_name, 1)
	# image, metadata, positions, label = data['image'], data['metadata'], data['positions'], data['label']

	# pos, pred_label = dc.detect(image, debug=True)

	# print(np.min(label), np.min(pred_label))

	# array_projection = np.max(image, axis=0)
	# label_projection = np.max(pred_label, axis=0)*255
	# sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	# sidebyside /= sidebyside.max()
	# plt.imsave('output/prediction.png', sidebyside)


	# exit()

	path = '/home/wahab/Data/HDD/Colloids/Real/Levke/Levke_smallParticles_betterData_2021_4_1/goodData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	array = dc.read_tif(path)
	print(array.shape)
	# array = array[:,0,:,:]
	array = array[0]

	pos, label = dc.detect(array, patch_overlap=(0,0,0), threshold=0.5, debug=True)
	print(pos.shape)
	dc.view(array, positions=pos, label=label)

	x, y = dc.get_gr(pos, 100, 100)
	plt.plot(x, y, label='pred')
	plt.show()

	# viewer=napari.Viewer()
	# viewer.add_image(array)
	# napari.run()	

	# import pdb;pdb.set_trace()