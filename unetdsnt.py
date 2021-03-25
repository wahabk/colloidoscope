import matplotlib.pyplot as plt
import numpy as np
import time
import segmentation_models_3D as sm
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
timestr = time.strftime("%Y-%m-%d-%H-%M")
from make_dataset import *
import tensorflow as tf
from generator import customImageDataGenerator


def lr_scheduler(epoch, learning_rate):
	decay_rate = 1
	decay_step = 8
	if epoch % decay_step == 0 and epoch not in [0, 1]:
		return learning_rate * decay_rate
	return learning_rate


def dataGenie(batch_size, data_gen_args, dataset = 'Test', n_samples=30):
	imagegen = customImageDataGenerator(**data_gen_args)
	maskgen = customImageDataGenerator(**data_gen_args)

	while True:
		scan_list, label_list = [], []
		for n in n_samples:

			scan = read_hdf5(dataset, n)
			label = read_hdf5(dataset+'_labels', n)
			scan_list.append(scan)
			label_list.append(label)        
		
		scan_list = np.array(scan_list, dtype='float32')
		label_list = np.array(label_list, dtype='float32')
		
		scan_list      = scan_list[:,:,:,:,np.newaxis] # add final axis to show datagens its grayscale
		label_list   = label_list[:,:,:,:,np.newaxis] # add final axis to show datagens its grayscale
		
		print(f'[dataGenie] Initialising image and mask generators \n Number of samples: {len(scan_list)}')
		image_generator = imagegen.flow(scan_list,
			batch_size = batch_size,
			#save_to_dir = 'output/Keras/',
			save_prefix = 'dataGenie',
			seed = 1,
			shuffle=True
			)
		mask_generator = maskgen.flow(label_list, 
			batch_size = batch_size,
			#save_to_dir = 'output/Keras/',
			save_prefix = 'dataGenie',
			seed = 1,
			shuffle=True
			)
		print('Ready.')

		datagen = zip(image_generator, mask_generator)
		for x_batch, y_batch in datagen:
			x_batch = x_batch/np.max(x_batch)
			y_batch = y_batch/np.max(y_batch)
			# print(x_batch[0].shape, x_batch[0].dtype, np.amax(x_batch[0]))
			# print(y_batch[0].shape, y_batch[0].dtype, np.amax(y_batch[0]))
			yield (x_batch, y_batch)


def testGenie(n, dataset = 'First'):
	scan, positions = read_hdf5(dataset, n, positions=True)
	labels = read_hdf5(f'{dataset}_labels', n, positions=False)
	scan = scan/255
	scan = scan[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
	labels = labels/255
	labels = labels[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
	return scan, positions, labels


if __name__ == "__main__":

	data_gen_args = dict(zoom_range=0.1,
						horizontal_flip=True,
						vertical_flip = True,
						fill_mode='constant',
						cval = 0)

	BACKBONE = 'resnet34'
	val_steps = 30
	batch_size = 1
	steps_per_epoch = 170
	epochs = 20
	nclasses = 1
	lr = 1e-5
	input_shape = [32,128,128,1]
	dataset = 'Simulated'
	n_samples = 30
	weightspath = 'output/Second.hdf5'
	activation = 'sigmoid'
	metrics = [sm.metrics.IOUScore(threshold=0.5)]

	model_checkpoint = ModelCheckpoint(weightspath, 
											monitor = 'loss', verbose = 1, save_best_only = True)

	callbacks = [
		keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
		model_checkpoint
	]

	datagenie = dataGenie(batch_size=batch_size,
							data_gen_args=data_gen_args,
							dataset=dataset,
							n_samples=range(1, 71))

	valdatagenie = dataGenie(batch_size=batch_size,
							data_gen_args=dict(),
							dataset=dataset,
							n_samples=range(71,101))



	optimizer = Adam(lr=lr)
	dice_loss = sm.losses.DiceLoss() 


	optimizer = Adam(learning_rate=lr)

	inp = sm.Unet(BACKBONE, input_shape=input_shape, encoder_weights=None, classes=nclasses, activation=activation, encoder_freeze=False)
	out = dsnt(inp)

	model = Model(inp, out, name=base_model.name)


	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

	# model.load_weights(weightspath)
	history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = steps_per_epoch, 
						epochs = epochs, callbacks=callbacks, validation_steps=val_steps)

	
	

	# model.load_weights(weightspath)

	# for n in range(1, 5):
	# test, positions = testGenie(1)
	# test_list.append(test)
	# position_list.append(positions)

	# test, positions, test_labels = testGenie(1, dataset=dataset)
	# print(test.shape)
	# print(test_labels.shape)
	# test_list = [test]
	# labels = model.predict(test_list, 1)
	# print('loss, acc:', loss, acc)

	
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(f'Unet loss (lr={lr})')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.ylim(0,1)
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(f'output/loss_curves/{timestr}_loss.png')
