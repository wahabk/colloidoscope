import h5py
import pandas as pd
import numpy as np

def read_x(path:str, index:int) -> np.ndarray:
	"""Read x from hdf 5 file at index

	Args:
		path (str): Path to hdf5 file.
		index (int): The index of the array to read.

	Returns:
		np.ndarray: 3D image array.
	"""
	with h5py.File(path, "r") as f:
		array = np.array(f[str(index)], dtype="uint8")
	return array

def read_y(path:str, index:int, dim_order:str="ZXY") -> np.ndarray:
	"""Read csv file with particle positions

	Args:
		path (str): Path to y data csv file.
		index (int): Index of positions to read.
		dim_order (str, optional): Dimension ordering. Defaults to "ZXY".

	Returns:
		np.ndarray: the positions of all the particles from the image.
	"""
	pos_df = pd.read_csv(path, index_col=0)
	pos_df = pos_df.dropna()
	pos_index = pos_df[pos_df.image_index == index]
	list_ = list(zip(pos_index[dim_order[0]], pos_index[dim_order[1]], pos_index[dim_order[2]]))
	positions = np.array(list_, dtype='float32')

	return positions

def extract_y(pos_df, index, dim_order="ZXY"):
	"""Helper function to extract indices without reading the csv repeatedly
	"""

	pos_df = pos_df.dropna()
	pos_index = pos_df[pos_df.image_index == index]
	list_ = list(zip(pos_index[dim_order[0]], pos_index[dim_order[1]], pos_index[dim_order[2]]))
	positions = np.array(list_, dtype='float32')

	return positions

def write_y(df, index, positions, dim_order="ZXY", min_n_pos=2000):
	"""Helper function to write a df

	Args:
		df (pandas.DataFrame): Dataframe to append the new positions to
		index (int): index of the image for the positions
		positions (np.ndarray): np.array of shape (n, 3) of particle positions
		dim_order (str, optional): _description_. Defaults to "ZXY".
		min_n_pos (int, optional): _description_. Defaults to 2000.

	Returns:
		_type_: _description_
	"""
	len_nans = min_n_pos - len(positions)
	nans = np.empty((len_nans, 3))
	nans[:] = np.nan
	positions = np.concatenate([positions, nans], axis=0)

	indices = np.full((len(positions),1), fill_value=int(index))
	positions_with_indices = np.concatenate([indices, positions], axis=1)
	pos_df = pd.DataFrame(positions_with_indices)
	dim1, dim2, dim3 = dim_order
	pos_df.columns = ["image_index", dim1, dim2, dim3]

	df = pd.concat([df, pos_df], axis=0, ignore_index=True)

	return df
