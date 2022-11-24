import h5py
import pandas as pd
import numpy as np

def read_x(path, index):
	with h5py.File(path, "r") as f:
		array = np.array(f[str(index)])
	return array

def read_y(path, index, dim_order="ZXY"):
	pos_df = pd.read_csv(path, index_col=0)
	pos_index = pos_df[pos_df.index == index]
	list_ = list(zip(pos_index[dim_order[0]], pos_index[dim_order[1]], pos_index[dim_order[2]]))
	positions = np.array(list_, dtype='float32')

	return positions

