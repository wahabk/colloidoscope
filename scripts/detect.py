import colloidoscope as cd
import napari
import matplotlib.pyplot as plt
import numpy as np
import scipy

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


	# path = '/home/wahab/Data/HDD/Colloids/Real/Levke/Levke_smallParticles_betterData_2021_4_1/goodData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	path = '/home/ak18001/Data/HDD/Colloids/Real/Levke/goodData_2021_4_1Levke_smallParticles_betterData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif'	
	array = dc.read_tif(path)
	print(array.shape)
	# array = array[:,0,:,:]
	array = array[0]
	# array = dc.crop3d(array, (32,128,128))

	pos, result = dc.detect(array, patch_overlap=(16,32,32), threshold=0.5, debug=True)
	print(pos.shape)
	# dc.view(array, positions=pos, label=result)

	x, y = dc.get_gr(pos, 100, 100)
	plt.plot(x, y, label='pred')
	# plt.show()

	# viewer=napari.Viewer()
	# viewer.add_image(array)
	# napari.run()	

	# import pdb;pdb.set_trace()