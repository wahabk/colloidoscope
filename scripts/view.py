from colloidoscope.deepcolloid import DeepColloid
from colloidoscope.predict import insert_in_center
import napari
import numpy as np
import matplotlib.pyplot as plt
import torch

def make_proj(test_array, test_label):

	array_projection = np.max(test_array, axis=0)
	label_projection = np.max(test_label, axis=0)*255

	if label_projection.shape != array_projection.shape:

		new_label = np.zeros_like(array_projection)
		a = array_projection.shape[0]
		l = label_projection.shape[0]
		diff = int((a-l)/2)
		print(a, l, diff)

		new_label[diff:a-diff, diff:a-diff] = label_projection
		label_projection = new_label
		

	sidebyside = np.concatenate((array_projection, label_projection), axis=1)
	# sidebyside /= sidebyside.max()

	return sidebyside

# dataset_path = '/home/wahab/Data/HDD/Colloids'
dataset_path = '/home/ak18001/Data/HDD/Colloids'
dc = DeepColloid(dataset_path)

dataset_name = 'r_width'

# sample, metadata, positions = dc.read_hdf5(dataset_name, 1, read_metadata=True)
# label, _ = dc.read_hdf5(dataset_name+'_labels', 1, read_metadata=False)
# nums = dc.get_hdf5_keys(dataset_name)

data = dc.read_hdf5(dataset_name, 2)
image, metadata, positions, label = data['image'], data['metadata'], data['positions'], data['label']

roi = (64,64,64)
blank = np.zeros_like(image)
image = torch.tensor(image)
blank = torch.zeros_like(image)

image = dc.crop3d(image, (60,60,60))

blank = insert_in_center(blank, image)

blank = blank.numpy()
blank = blank.max(axis=0)


# sidebyside = make_proj(image, label)

plt.imsave('output/test.png', blank)

# print(image.shape, image.max(), image.min())
# print(label.shape, label.max(), label.min())

# dc.view(image, positions, label)
# napari.run()