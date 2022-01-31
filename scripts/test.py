import colloidoscope

if __name__ == '__main__':
    dataset_path = '/Users/wahab/Data/Colloids/'

    dc = colloidoscope.DeepColloid(dataset_path)
    
    array = dc.read_tif('Real/Levke/goodData_2021_4_1Levke_smallParticles_betterData_2021_4_1/Levke_smallParticlesL1S_31_dense_1_4_21_Series006.tif')

    print(array.shape)

    dc.view(array)