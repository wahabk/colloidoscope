import torch
import numpy as np
from colloidoscope.deepcolloid import DeepColloid
import matplotlib.pyplot as plt



if __name__ == "__main__":
    dataset_path = '/home/ak18001/Data/HDD/Colloids'
    # dataset_path = '/home/wahab/Data/HDD/Colloids'
    # dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
    dc = DeepColloid(dataset_path)
    dataset_name = 'new_year'

    test_array, metadata, true_positions = dc.read_hdf5(dataset_name, 40, read_metadata=True)
    print(metadata)
    x, y = dc.get_gr(true_positions, 7, 25)
    print(x,y)
    plt.plot(x, y, label='true') 

    test_array, metadata, true_positions = None, None, None

    test_array, metadata, true_positions = dc.read_hdf5(dataset_name, 50, read_metadata=True)
    print(metadata)
    x, y = dc.get_gr(true_positions, 7, 25)
    plt.plot(x, y, label='pred')

    plt.savefig('output/gr.png')

