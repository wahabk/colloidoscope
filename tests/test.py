from colloidoscope import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari

if __name__ == '__main__':
	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	dataset = 'feb_psf_const'
	n=1

	nums = dc.get_hdf5_keys(dataset)
	print(type(nums), nums)

	data = dc.read_hdf5(dataset, n)
	array, label, positions = data['image'], data['label'], data['positions']

	print(array.shape, label.shape, positions.shape)

	dc.view(array, positions=positions, label=label)
	# napari.run()

