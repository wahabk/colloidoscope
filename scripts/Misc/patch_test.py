import numpy as np

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



class sampler():
	def __init__(self, image:np.ndarray, patch_size:np.ndarray, overlap:np.ndarray, in_size:np.ndarray,) -> None:
		self.image = image
		self.patch_size = patch_size
		self.overlap = overlap
		self.in_size = in_size
		self.needs_padding = False
		pass

	def __len__(self) -> int:
		pass

class aggregator():
	def __init__(self, sampler:sampler, ) -> None:
		self.sampler = sampler
		self.batches = []
		self.locations = []
		pass

	def append_batch(self, batch, loc) -> None:
		self.batches.append(batch)
		self.locations.append(loc)

	def get_output(self,) -> np.ndarray:
		pass




locs = get_patches_locations([130,130,64], [64,64,64], [16,16,16])


print(locs)