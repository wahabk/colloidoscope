import colloidoscope as cd
import napari

if __name__ == '__main__':
	dataset_path = dataset_path = '/home/wahab/Data/HDD/Colloids/'

	dc = cd.DeepColloid(dataset_path)
	path = '/home/wahab/Data/HDD/Colloids/Real/Levke/Levke_smallParticles_betterData_2021_4_1/goodData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	array = dc.read_tif(path)

	array = array[0]	
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

	print(array.shape)

	# pos, label = dc.detect(array, debug=True)
	# print(pos.shape)
	# dc.view(array, positions=pos, label=label)

	viewer=napari.Viewer()
	viewer.add_image(array)
	napari.run()	

	import pdb;pdb.set_trace()