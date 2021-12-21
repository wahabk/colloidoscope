from colloidoscope.deepcolloid import DeepColloid
import napari

dataset_path = '/home/ak18001/Data/HDD/Colloids'

dc = DeepColloid(dataset_path)

sample, position = dc.read_hdf5('replicate_labels', 1, return_positions=True)
nums = dc.get_hdf5_keys('third_run')
print(nums)


dc.view(sample)
napari.run()