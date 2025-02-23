import torch
import numpy as np
from colloidoscope.train_utils import Trainer, LearningRateFinder, test, ColloidsDatasetSimulated
from colloidoscope.deepcolloid import DeepColloid
import torchio as tio
from neptune.new.types import File
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
from ray import tune
import random
import copy
import monai
import math
from pathlib2 import Path
from monai.networks.layers.factories import Act, Norm

print(os.cpu_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.is_available())
print('------------num available devices:', torch.cuda.device_count())

import torch.nn as nn
import torch.nn.functional as F


def train(config, name, dataset_path, dataset_name, train_data, val_data, test_data, save=False, 
			tuner=True, device_ids=[0,1], num_workers=10, work_dir=None, post_processing="tp"):
	os.chdir(work_dir)
	'''
	by default for ray tune
	'''

	#TODO calculate nblocks or only pad first block

	dc = DeepColloid(dataset_path)

	# setup neptune
	run = neptune.init(
		project="wahabk/colloidoscope",
		# api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
		api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMzZlNGZhMi1iMGVkLTQzZDEtYTI0MC04Njk1YmJmMThlYTQifQ==",
	)
	params = dict(
		roiSize = (100,100,100),
		train_data = train_data,
		val_data = val_data,
		test_data = test_data,
		dataset_name = dataset_name,
		batch_size = config['batch_size'],
		n_blocks = config['n_blocks'],
		norm = config['norm'],
		loss_function = config['loss_function'],
		lr = config['lr'],
		epochs = config['epochs'],
		start_filters = config['start_filters'],
		activation = config['activation'],
		dropout = config['dropout'],
		num_workers = num_workers,
		n_classes = 1,
		random_seed = 42,
	)
	run['Tags'] = name
	
	#TODO find a way to precalculate this - should i only unpad the first block?
	# if config['n_blocks'] == 2: label_size = (48,48,48)
	# if config['n_blocks'] == 3: label_size = (24,24,24)
	# label_size = params['roiSize']
	label_size = [96,96,96]

	transforms_affine = tio.Compose([
		tio.RandomFlip(axes=(0,1,2), flip_probability=0.5),
		# tio.RandomAffine(), # TODO add random affine for crop and bc
	])
	transforms_img = tio.Compose([
		tio.RandomAnisotropy(p=0.1),              # make images look anisotropic 25% of times
		tio.RandomBlur(p=0.1),
		tio.OneOf({
			tio.RandomNoise(0.1, 0.01): 0.1,
			tio.RandomBiasField(0.1): 0.1,
			tio.RandomGamma((-0.3,0.3)): 0.1,
			tio.RandomMotion(): 0.3,
		}),
		tio.RescaleIntensity((0.05,0.95)),
	])

	# create a training data loader
	train_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=transforms_img, label_transform=None, label_size=label_size) 
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
	# create a validation data loader
	val_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['val_data'], label_size=label_size) 
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())

	# device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'training on {device}')

	start_filters = params['start_filters']
	n_blocks = params['n_blocks']
	start = int(math.sqrt(start_filters))
	channels = [2**n for n in range(start, start + n_blocks)]
	strides = [2 for n in range(1, n_blocks)]

	print(f"channels {channels}, strides {strides}")

	# model = monai.networks.nets.UNet(
	# 	spatial_dims=3,
	# 	in_channels=1,
	# 	out_channels=params['n_classes'],
	# 	channels=channels,
	# 	strides=strides,
	# 	num_res_units=params["n_blocks"],
	# 	act=params['activation'],
	# 	norm=params["norm"],
	# 	dropout=params["dropout"]
	# )

	model = monai.networks.nets.AttentionUnet(
		spatial_dims=3,
		in_channels=1,
		out_channels=params['n_classes'],
		channels=channels,
		strides=strides,
		kernel_size=7,
		# up_kernel_size=3,
		dropout=params["dropout"],
		padding='valid',
	)

	# model = monai.networks.nets.UNETR(
	# 	in_channels = 1,
	# 	out_channels = 1,
	# 	img_size=params['roiSize'],
	# 	feature_size=32,
	# 	hidden_size=768,
	# 	mlp_dim=3072,
	# 	num_heads=12,
	# 	pos_embed='conv', 
	# 	norm_name='instance', 
	# 	conv_block=True, 
	# 	res_block=True, 
	# 	dropout_rate=0.0, 
	# 	spatial_dims=3, 
	# 	qkv_bias=False)

	# model = monai.networks.nets.ViT(
	# 	in_channels=1,
	# 	img_size=params['roiSize'],
	# 	patch_size=params['roiSize'],
	# 	hidden_size=768,
	# 	mlp_dim=3072,
	# 	num_layers=12,
	# 	num_heads=12,
	# 	pos_embed='conv',
	# 	dropout_rate=0.0,
	# 	spatial_dims=3,)

	"""
	if padding mode is valid use roisize+/-4
	if padding is same, use roisize
	"""

	# summary(model, (8,1,68,68,68))

	# model
	# model = UNet(in_channels=1,
	# 			out_channels=params['n_classes'],
	# 			n_blocks=params['n_blocks'],
	# 			start_filters=params['start_filters'],
	# 			activation=params['activation'],
	# 			normalization=params['norm'],
	# 			conv_mode='valid',
	# 			up_mode='transposed',
	# 			dim=3,
	# 			skip_connect=None,
	# 			)

	model = torch.nn.DataParallel(model, device_ids=device_ids)
	model.to(device)

	# loss function
	criterion = params['loss_function']
	params['loss_function'] = str(copy.deepcopy(params['loss_function']))

	if isinstance(criterion, monai.losses.FocalLoss):
		params['gamma'] = criterion.gamma

	# optimizer
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	optimizer = torch.optim.Adam(model.parameters(), params['lr'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
	# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, cycle_momentum=False)

	run['parameters'] = params

	# trainer
	trainer = Trainer(model=model,
					device=device,
					criterion=criterion,
					optimizer=optimizer,
					training_DataLoader=train_loader,
					validation_DataLoader=val_loader,
					lr_scheduler=scheduler,
					epochs=params['epochs'],
					logger=run,
					tuner=tuner,
					transformer=False,
					)

	# start training
	training_losses, validation_losses, lr_rates = trainer.run_trainer()

	run['learning_rates'].log(lr_rates)
	
	if save:
		model_name = save
		torch.save(model.state_dict(), model_name)
		run['model/weights'].upload(model_name)

	losses = test(model, dataset_path, dataset_name, test_data, run=run, 
				criterion=criterion, device=device, num_workers=num_workers, batch_size=1,
				canvas_size=params['roiSize'], label_size=label_size, heatmap_r='radius', 
				work_dir=work_dir, post_processing=post_processing)
	run['test/df'].upload(File.as_html(losses))

	run.stop()

if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/scratch/ak18001/Colloids/'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	# dataset_path = '/home/wahab/Data/HDD/Colloids'
	dc = DeepColloid(dataset_path)

	dataset_name = 'please_1400'
	n_samples = dc.get_hdf5_keys(dataset_name)
	print(len(n_samples))
	all_data = list(range(1,1400))
	test_data =	list(range(1,599))
	random.shuffle(all_data)
	random.shuffle(test_data)

	# train_data = all_data[0:1200]
	# val_data = all_data[1200:1400]
	# test_data = test_data[:200]
	train_data = all_data[0:100]
	val_data = all_data[100:150]
	test_data = test_data[:50]
	name = 'please sharkbait'
	# save = 'output/weights/attention_unet_202206.pt'
	# save = '/user/home/ak18001/scratch/Colloids/attention_unet_20220524.pt'
	save = False
	post_processing = "tp"

	print(f"training on {len(train_data)} val {len(val_data)} test {len(test_data)}")


	config = {
		"lr": 0.002165988, #0.00122678,
		"batch_size": 2,
		"n_blocks": 3,
		"norm": 'INSTANCE',
		"epochs": 1,
		"start_filters": 32,
		"activation": "SWISH",
		"dropout": 0,
		"loss_function": torch.nn.L1Loss(),
		# "loss_function":	monai.losses.FocalLoss(gamma=0.5)#torch.nn.BCEWithLogitsLoss()	# BinaryFocalLoss(alpha=1.5, gamma=0.5) # torch.nn.BCEWithLogitsLoss() #torch.nn.L1Loss() #  #,
	}


	work_dir = Path().parent.resolve()
	train(config, name, dataset_path=dataset_path, dataset_name=dataset_name, 
			train_data=train_data, val_data=val_data, test_data=test_data, 
			save=save, tuner=False, device_ids=[0,], work_dir=work_dir, post_processing=post_processing)
