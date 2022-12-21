import colloidoscope
from read import read_x, read_y
import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np
from tqdm import tqdm

def write_y(big_d, positions, dim_order="ZXY", 	min_n_pos=2000):
	len_nans = min_n_pos - len(positions)
	nans = np.empty((len_nans, 3))
	nans[:] = np.nan
	positions = np.concatenate([positions, nans], axis=0)

	indices = np.full((len(positions),1), fill_value=int(index))
	positions_with_indices = np.concatenate([indices, positions], axis=1)
	pos_df = pd.DataFrame(positions_with_indices)
	dim1, dim2, dim3 = dim_order
	pos_df.columns = ["image_index", dim1, dim2, dim3]

	big_d = pd.concat([big_d, pos_df], axis=0, ignore_index=True)

	return big_d

if __name__ == '__main__':
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/home/ak18001/Data/HDD/Colloids/challenge/'
	dataset_path = '/home/wahab/Data/HDD/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = colloidoscope.DeepColloid(dataset_path)

	from read import read_x, read_y
	from metric import average_precision, plot_pr, remove_borders
	
	path = ""
	x_path = dataset_path+"challenge/x_test.hdf5"
	m_path = dataset_path+"challenge/x_test_metadata.csv"
	y_path = dataset_path+"challenge/y_test.csv"
	y_path_new = dataset_path+"challenge/y_test_new.csv"

	pos_df = pd.read_csv(y_path)
	dim_order = "ZXY"
	print(pos_df)

	# MIN N LENGTH FROM TEST

	# lengths = []
	# for index in tqdm(range(10)):
	# 	pos_index = pos_df[pos_df.image_index == index]
	# 	list_ = list(zip(pos_index[dim_order[0]], pos_index[dim_order[1]], pos_index[dim_order[2]]))
	# 	pos = np.array(list_, dtype='float32')
	# 	length = len(pos)
	# 	lengths.append(length)
	# 	# print(index, length)
	# longest = max(lengths)
	# print(longest)
		
	big_d = pd.DataFrame(columns=['image_index', "Z", "X", "Y"])

	for index in tqdm(range(10)):
		pos_index = pos_df[pos_df.image_index == index]
		list_ = list(zip(pos_index[dim_order[0]], pos_index[dim_order[1]], pos_index[dim_order[2]]))
		positions = np.array(list_, dtype='float32')

		big_d = write_y(big_d, positions)

	big_d.to_csv(y_path_new)

