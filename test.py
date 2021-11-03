from src.deepcolloid import DeepColloid
import numpy as np
import matplotlib.pyplot as plt
import napari

if __name__ == '__main__':
	dc = DeepColloid('/home/wahab/Data/HDD/Colloids')

	dataset = 'First'
	n=1

	nums = dc.get_hdf5_keys(dataset)

	print(type(nums), nums)
	scan, positions = dc.read_hdf5(dataset, n, return_positions=True)
	label = dc.read_hdf5(dataset+'_labels', n)

	print(scan.shape, label.shape)

	dc.view(scan, positions)
	napari.run()

