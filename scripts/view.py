from colloidoscope.deepcolloid import DeepColloid
import napari
import numpy as np

dataset_path = '/home/ak18001/Data/HDD/Colloids'
dc = DeepColloid(dataset_path)

dataset_name = 'new_year'

# TODO find consecutive time points and view
sample, metadata, positions = dc.read_hdf5(dataset_name, 1, read_metadata=True)
label, _ = dc.read_hdf5(dataset_name+'_labels', 1, read_metadata=False)
nums = dc.get_hdf5_keys(dataset_name)

dc.view(sample, positions, label)
napari.run()