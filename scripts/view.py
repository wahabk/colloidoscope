from colloidoscope.deepcolloid import DeepColloid
import napari
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/home/wahab/Data/HDD/Colloids/'
# dataset_path = '/home/ak18001/Data/HDD/Colloids'
dc = DeepColloid(dataset_path)

dataset_name = 'feb_final'

# sample, metadata, positions = dc.read_hdf5(dataset_name, 1, read_metadata=True)
# label, _ = dc.read_hdf5(dataset_name+'_labels', 1, read_metadata=False)
# nums = dc.get_hdf5_keys(dataset_name)

data = dc.read_hdf5(dataset_name, 2)
image, metadata, positions, label = data['image'], data['metadata'], data['positions'], data['label']

# array_projection = np.max(image, axis=0)
# label_projection = np.max(label, axis=0)*255
# sidebyside = np.concatenate((array_projection, label_projection), axis=1)
# sidebyside /= sidebyside.max()
# plt.imsave('output/test.png', sidebyside)

print(image.shape, image.max(), image.min())
print(label.shape, label.max(), label.min())

dc.view(image, positions, label)
# napari.run()