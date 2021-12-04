from colloidoscope import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari

if __name__ == '__main__':
	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)

	dataset = 'replicate'
	n=1

	nums = dc.get_hdf5_keys(dataset)

	print(type(nums), nums)
	scan, positions = dc.read_hdf5(dataset, n, return_positions=True)
	label = dc.read_hdf5(dataset+'_labels', n)

	print(scan.shape, label.shape)

	# dc.view(scan, positions)
	# napari.run()

