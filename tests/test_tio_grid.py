import numpy as np
import torch
import torchio as tio

from colloidoscope import DeepColloid



if __name__ == "__main__":

	# dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dc = DeepColloid(dataset_path)

	roiSize = (1,64,64,64)

	array = np.zeros(roiSize)

	subject_dict = {'array' : tio.ScalarImage(tensor=array, type=tio.INTENSITY, path=None),}
	subject = tio.Subject(subject_dict) 
	grid_sampler = tio.inference.GridSampler(subject, patch_size=(32,32,32), patch_overlap=(16,16,16), padding_mode='mean')
	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
	aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop') 


	for i, patch_batch in enumerate(patch_loader):
		tensor = patch_batch['array'][tio.DATA]
		locations = patch_batch[tio.LOCATION]

		aggregator.add_batch(tensor, locations)

	output_tensor = aggregator.get_output_tensor()
	print(output_tensor.shape)

