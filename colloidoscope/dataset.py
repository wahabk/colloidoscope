import numpy as np
import torch
from .deepcolloid import DeepColloid

class ColloidsDatasetSimulated(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for simulated colloids

	transform is augmentation function

	"""	

	def __init__(self, dataset_path:str, dataset_name:str, indices:list, transform=None, label_transform=None):	
		super().__init__()
		self.dataset_path = dataset_path
		self.dataset_name = dataset_name
		self.indices = indices
		self.transform = transform
		self.label_transform = label_transform


	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		dc = DeepColloid(self.dataset_path)
		# Select sample
		i = self.indices[index]

		X, metadata, positions = dc.read_hdf5(self.dataset_name, i)
		# TODO make arg return_positions = True for use in DSNT
		y, positions = dc.read_hdf5(self.dataset_name+'_labels', i, read_metadata=False)



		# dc.view(X)
		# napari.run()

		X = np.array(X/X.max(), dtype=np.float32)
		y = np.array(y/y.max() , dtype=np.float32)
		
		# print('x', np.min(X), np.max(X), X.shape)
		# print('y', np.min(y), np.max(y), y.shape)

		#fopr reshaping"
		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor

		if self.transform:
			X, y = self.transform(X), self.label_transform(y)
		# if self.label_transform:
		# 	y = self.label_transform(X)

		# print('x', np.min(X), np.max(X), X.shape)
		# print('y', np.min(y), np.max(y), y.shape)

		del dc
		return X, y


def compute_max_depth(shape= 1920, max_depth=10, print_out=True):
    shapes = []
    shapes.append(shape)
    for level in range(1, max_depth):
        if shape % 2 ** level == 0 and shape / 2 ** level > 1:
            shapes.append(shape / 2 ** level)
            if print_out:
                print(f'Level {level}: {shape / 2 ** level}')
        else:
            if print_out:
                print(f'Max-level: {level - 1}')
            break

	#out = compute_max_depth(shape, print_out=True, max_depth=10)
    return shapes