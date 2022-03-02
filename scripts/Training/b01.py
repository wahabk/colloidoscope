import torch
import numpy as np
from colloidoscope.trainer import Trainer, LearningRateFinder, predict, test, train
from colloidoscope.unet import UNet
from colloidoscope.dataset import ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
import torchio as tio
import matplotlib.pyplot as plt
import neptune.new as neptune
from neptune.new.types import File
import os
from ray import tune
import random

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)

	dataset_name = 'march_first'
	n_samples = dc.get_hdf5_keys(dataset_name)
	print(len(n_samples))
	all_data = list(range(2,3000))
	random.shuffle(all_data)

	train_data = all_data[0:600]
	val_data = all_data[601:800]
	test_data =	all_data[801:900]
	name = 'trying march dice'
	save = 'output/weights/unet.pt'
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'

	config = {
		"lr": 0.0001,
		"batch_size": 4,
		"n_blocks": 5,
		"norm": 'batch',
		"epochs": 10,
		"start_filters": 32,
		"activation": 'relu',
		"loss_function": torch.nn.L1Loss(),
	}

	train(config, name, dataset_path=dataset_path, dataset_name=dataset_name, 
				train_data=train_data, val_data=val_data, test_data=test_data, 
				save=save, tuner=False, device_ids=[0,])
