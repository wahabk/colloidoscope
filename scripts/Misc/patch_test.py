import numpy as np
import colloidoscope

import torch
import numpy as np
import warnings
from typing import Optional, Tuple, Generator, Union

import scipy
import torch
import numpy as np

import trackpy as tp
import pandas as pd
import torchio as tio
import monai
from tqdm import tqdm
from pathlib2 import Path

import torch.nn.functional as F

import copy

# from patchify import patchify, unpatchify
# image = np.zeros((128,128,128))
# patch_size = (64,64,64)
# patches = patchify(image, patch_size, 64)
# print(patches.shape)
# recon = unpatchify(patches, image.shape)
# print(recon.shape)

def get_patches_locations( image_size:list, patch_size:list, patch_overlap:list,) -> np.ndarray:
		# Example with image_size 10, patch_size 5, overlap 2:
		# [0 1 2 3 4 5 6 7 8 9]
		# [0 0 0 0 0]
		#       [1 1 1 1 1]
		#           [2 2 2 2 2]
		# Locations:
		# [[0, 5],
		#  [3, 8],
		#  [5, 10]]
		indices = []
		zipped = zip(image_size, patch_size, patch_overlap)
		for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
			end = im_size_dim + 1 - patch_size_dim
			step = patch_size_dim - patch_overlap_dim
			indices_dim = list(range(0, end, step))
			if indices_dim[-1] != im_size_dim - patch_size_dim:
				indices_dim.append(im_size_dim - patch_size_dim)
			print(indices_dim)
			indices.append(indices_dim)
		indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
		indices_ini = np.unique(indices_ini, axis=0)
		indices_fin = indices_ini + np.array(patch_size)
		locations = np.hstack((indices_ini, indices_fin))
		return np.array(sorted(locations.tolist()))

class MyGridSampler(torch.utils.data.Dataset):
	def __init__(self, image:torch.tensor, patch_size:list, overlap:list=[0,0,0], label_size:Union[list, None]=None,) -> None:
		"""
		based on tio
		"""
		self.image = image
		self.patch_size = patch_size
		self.overlap = overlap
		self.label_size = label_size

		self.patch_tensor, self.label_smaller = self._make_patch_tensor(image, 
																		np.array(patch_size), 
																		np.array(overlap), 
																		np.array(label_size))
		
		self.locations = self._get_patches_locations(self.patch_tensor.shape[1:], self.patch_size, patch_overlap=self.overlap)

		if self.label_smaller: self.label_locations = self._get_patches_locations(self.image.shape[1:], self.label_size, self.overlap)
		else: self.label_locations = self.locations

	def __len__(self) -> int:
		return len(self.locations)

	def __getitem__(self, index):
		# Assume 3D
		location = copy.deepcopy(torch.tensor(self.locations[index]))
		index_ini = location[:3]
		print("getting", index, index_ini, )
		cropped = self.crop(self.patch_tensor, index_ini, self.patch_size)
		label_location = torch.tensor(self.label_locations[index])
		d = {"ims":cropped,"loc":label_location}
		return d

	def _make_patch_tensor(self, image:torch.tensor, patch_size:np.ndarray, overlap:np.ndarray, label_size:np.ndarray,):

		if label_size is None:
			patch_tensor = image
			label_smaller = False
			raise NotImplementedError("")
			pass
		else: 
			diffs = patch_size - label_size
			# tensor_size = np.array(image.shape) + ([0]+list(diffs*2))
			patch_tensor = F.pad(self.image, (diffs[0],diffs[0],diffs[1],diffs[1],diffs[2],diffs[2]))
			label_smaller = True

		return patch_tensor, label_smaller

	def crop(self, image, index_ini, patch_size):
			return image[	:,
				index_ini[0]:index_ini[0]+patch_size[0], 
				index_ini[1]:index_ini[1]+patch_size[1], 
				index_ini[2]:index_ini[2]+patch_size[2]]


	def _get_patches_locations(self, image_size:list, patch_size:list, patch_overlap:list,) -> np.ndarray:
		# Example with image_size 10, patch_size 5, overlap 2:
		# [0 1 2 3 4 5 6 7 8 9]
		# [0 0 0 0 0]
		#       [1 1 1 1 1]
		#           [2 2 2 2 2]
		# Locations:
		# [[0, 5],
		#  [3, 8],
		#  [5, 10]]
		indices = []
		zipped = zip(image_size, patch_size, patch_overlap)
		for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
			end = im_size_dim + 1 - patch_size_dim
			step = patch_size_dim - patch_overlap_dim
			indices_dim = list(range(0, end, step))
			if indices_dim[-1] != im_size_dim - patch_size_dim:
				indices_dim.append(im_size_dim - patch_size_dim)
			indices.append(indices_dim)
		indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
		indices_ini = np.unique(indices_ini, axis=0)
		indices_fin = indices_ini + np.array(patch_size)
		locations = np.hstack((indices_ini, indices_fin))
		return np.array(sorted(locations.tolist()))








class MyAggregator():
	def __init__(self, sampler:MyGridSampler, overlap_mode="avg") -> None:
		self.sampler = sampler
		self.overlap_mode = overlap_mode

		self._out_tensor = torch.zeros(self.sampler.image.shape) # remember to add batch dim
		if overlap_mode=='avg': 
			self.avgmask_tensor = torch.zeros_like(self._out_tensor)


	def append_batch(self, batch_tensor, loc) -> None:

		batch = batch_tensor.cpu()
		locs = loc.cpu().numpy()

		if self.overlap_mode == "avg":
			for patch, location in zip(batch, locs):
				i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
				self._out_tensor[
					:,
					i_ini:i_fin,
					j_ini:j_fin,
					k_ini:k_fin,
				] += patch
				self.avgmask_tensor[
					:,
					i_ini:i_fin,
					j_ini:j_fin,
					k_ini:k_fin,
				] += 1
		else: raise NotImplementedError("")


	def get_output(self,) -> torch.tensor:
		if self.sampler.label_smaller: return self._out_tensor / self.avgmask_tensor
		else: raise NotImplementedError("")



# image_size = [208,208,208]
# patch_size = [100,100,100]
# overlap_size = [0,0,0]

# locs = get_patches_locations(image_size, patch_size, overlap_size)
# print(len(locs))
# # print(locs)

# image_size = [200,200,200]
# patch_size = [96,96,96]
# overlap_size = [0,0,0]

# locs = get_patches_locations(image_size, patch_size, overlap_size)
# print(len(locs))
# # print(locs)

dc = colloidoscope.DeepColloid()

image_size = [1,128,128,128]
patch_size = [64,64,64]
label_size = [60,60,60]
overlap_size = [0,0,0]
batch_size = 1
array = dc.read_tif('examples/Data/emily.tiff')
array = dc.crop3d(array, roiSize=image_size[1:])
array = np.expand_dims(array, 0)
t = torch.from_numpy(array)
print(t.shape)

grid_sampler = MyGridSampler(t, patch_size=patch_size, overlap=overlap_size, label_size=label_size)
patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
aggregator = MyAggregator(grid_sampler, overlap_mode='avg') # average for bc, crop for normal

for batch_dict in patch_loader:
	# print(batch_dict["ims"].shape)
	print(batch_dict["loc"])

	img = batch_dict["ims"]
	locs = batch_dict["loc"]

	img = img[	:,
				:,
				2:62,
				2:62,
				2:62]
	# print(img.shape)

	aggregator.append_batch(img, locs)

tensor = aggregator.get_output()
array = tensor.cpu().numpy()
array = np.squeeze(array)

dc.view(array)
