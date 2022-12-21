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

