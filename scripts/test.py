import colloidoscope

if __name__ == '__main__':
    dataset_path = '/Users/wahab/Data/Colloids/'

    dc = colloidoscope.DeepColloid(dataset_path)
    
    array = dc.read_tif('Real/Levke/goodData_2021_4_1Levke_smallParticles_betterData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif')

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

    dc.view(array)