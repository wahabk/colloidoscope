# Copyright 2020 MONAI Consortium

import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, RandRotate90, RandSpatialCrop, ScaleIntensity, EnsureType
from monai.visualize import plot_2d_or_3d_image
from src.deepcolloid import DeepColloid
print(torch.cuda.is_available())
import napari

class ColloidsDatasetSimulated(torch.utils.data.Dataset):
	"""
	
	Torch Dataset for simulated colloids

	transform is augmentation function

	"""	

	def __init__(self, dataset_path:str, dataset_name:str, indices:list, transform=None, label_transform=None):
		super().__init__()
		self.dataset_path = dataset_path
		self.dataset_name = dataset_name
		self.indices = indices
		self.transform = transform
		self.label_transform = label_transform


	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		dc = DeepColloid(self.dataset_path)
		# Select sample
		i = self.indices[index]

		X, positions = dc.read_hdf5(self.dataset_name, i, return_positions=True)
		# TODO make arg return_positions = True for use in DSNT
		y = dc.read_hdf5(self.dataset_name+'_labels', i)

		# dc.view(X)
		# napari.run()

		X = np.array(X/255, dtype=np.float32)
		y = np.array(y , dtype=np.float32)
		# print('x', np.min(X), np.max(X), X.shape)
		# print('y', np.min(y), np.max(y), y.shape)


		#fopr reshaping"
		X = np.expand_dims(X, 0)      # if numpy array
		y = np.expand_dims(y, 0)
		# tensor = tensor.unsqueeze(1)  # if torch tensor

		if self.transform:
			X, y = self.transform(X), self.label_transform(y)
		# if self.label_transform:
		# 	y = self.label_transform(X)



		del dc
		return X, y


def compute_max_depth(shape= 1920, max_depth=10, print_out=True):
    shapes = []
    shapes.append(shape)
    for level in range(1, max_depth):
        if shape % 2 ** level == 0 and shape / 2 ** level > 1:
            shapes.append(shape / 2 ** level)
            if print_out:
                print(f'Level {level}: {shape / 2 ** level}')
        else:
            if print_out:
                print(f'Max-level: {level - 1}')
            break

	#out = compute_max_depth(shape, print_out=True, max_depth=10)
    return shapes


if __name__ == '__main__':
	
	# monai.config.print_config()
	# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	dataset_path = '/home/wahab/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	dc = DeepColloid(dataset_path)
	# dc = DeepColloid(dataset_path)

	roiSize = (32,128,128)
	train_data = range(1,9)
	val_data = range(9,11)
	dataset_name = 'replicate'
	batch_size = 2
	num_workers = 2
	epochs=10
	n_classes=1
	lr = 3e-5


	# define transforms for image and segmentation
	train_imtrans = Compose([
		# AddChannel(),
		ScaleIntensity(),
		# RandRotate90(prob=0.5, spatial_axes=(0, 2)),
		EnsureType(),
	])
	train_segtrans = Compose([
		# AddChannel(),
		# RandRotate90(prob=0.5, spatial_axes=(0, 2)),
		EnsureType(),
	])
	val_imtrans = Compose([
		ScaleIntensity(), 
		# AddChannel(), 
		EnsureType(),
		])
	val_segtrans = Compose([
		# AddChannel(), 
		EnsureType(),
	])

	# define image dataset, data loader
	check_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=train_imtrans, label_transform=train_segtrans) 
	check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
	im, seg = monai.utils.misc.first(check_loader)
	print(im.shape, seg.shape)

	# create a training data loader
	train_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, train_data, transform=train_segtrans, label_transform=train_segtrans) 
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
	# create a validation data loader
	val_ds = ColloidsDatasetSimulated(dataset_path, dataset_name, val_data, transform=val_imtrans, label_transform=val_segtrans) 
	val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
	dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
	post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

	# create UNet, DiceLoss and Adam optimizer
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('-'*10)
	print(f'training on {device}')
	model = monai.networks.nets.UNet(
		spatial_dims=3,
		in_channels=1,
		out_channels=n_classes,
		channels=(16, 32, 64, 128, 256),
		strides=(2, 2, 2, 2),
		num_res_units=2,
		# act=torch.nn.activation.ReLU(),
	).to(device)
	# loss_function = monai.losses.DiceLoss(sigmoid=True)
	loss_function = torch.nn.BCEWithLogitsLoss()
	
	optimizer = torch.optim.Adam(model.parameters(), lr)

	# start a typical PyTorch training
	val_interval = 2
	best_metric = -1
	best_metric_epoch = -1
	epoch_loss_values = list()
	metric_values = list()
	writer = SummaryWriter()
	for epoch in range(epochs):
		print("-" * 10)
		print(f"epoch {epoch + 1}/{5}")
		model.train()
		epoch_loss = 0
		step = 0
		for batch_data in train_loader:
			step += 1
			inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
			# print('DataSet shape: ', inputs.shape, labels.shape)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_function(outputs, labels)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			epoch_len = len(train_ds) // train_loader.batch_size
			print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
			writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
		epoch_loss /= step
		epoch_loss_values.append(epoch_loss)
		print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

		if (epoch + 1) % val_interval == 0:
			model.eval()
			with torch.no_grad():
				val_images = None
				val_labels = None
				val_outputs = None
				for val_data in val_loader:
					val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
					sw_batch_size = 2
					val_outputs = sliding_window_inference(val_images, roiSize, sw_batch_size, model)
					val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
					# compute metric for current iteration
					dice_metric(y_pred=val_outputs, y=val_labels)
				# aggregate the final mean dice result
				metric = dice_metric.aggregate().item()
				# reset the status for next validation round
				dice_metric.reset()
				metric_values.append(metric)
				if metric > best_metric:
					best_metric = metric
					best_metric_epoch = epoch + 1
					torch.save(model.state_dict(), "best_metric_model_segmentation3d_array.pth")
					print("saved new best metric model")
				print(
					"current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
						epoch + 1, metric, best_metric, best_metric_epoch
					)
				)
				writer.add_scalar("val_mean_dice", metric, epoch + 1)
				# plot the last model output as GIF image in TensorBoard with the corresponding image and label
				plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
				plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
				plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

	print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
	writer.close()
