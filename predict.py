import matplotlib.pyplot as plt
import numpy as np
import time
import segmentation_models_3D as sm
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, schedules
timestr = time.strftime("%Y-%m-%d-%H-%M")
from make_dataset import *
import tensorflow as tf
from mainviewer import mainViewer
from unet import dataGenie
import trackpy

def testGenie(n, dataset):
	scan = read_hdf5(dataset, n)
	scan = scan/np.max(scan)
	scan = scan[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
	return scan


BACKBONE = 'resnet34'
val_steps = 19
batch_size = 1
steps_per_epoch = 80
epochs = 40
nclasses = 1
lr = 1e-5
input_shape = [32,128,128,1]
dataset = 'Real'
n_samples = 30
weightspath = 'output/unet_checkpoints2.hdf5'
activation = 'sigmoid'



model = sm.Unet(BACKBONE, input_shape=input_shape, encoder_weights=None, classes=nclasses, activation=activation, encoder_freeze=False)

model.load_weights(weightspath)
test_list = []
position_list = []

for n in range(1,400, 50):
	test = testGenie(n, dataset)
	test_list.append(test)

test_list = np.array(test_list)
print(test_list.shape)


results = model.predict(test_list, batch_size=1) # read about this one

for i, result in enumerate(results):
	result = results[i]*255
	test = test_list[i]*255
	print(result.shape, np.max(result), np.mean(result), np.min(result))
	result = np.squeeze(result.astype('uint8'), axis = 3)
	test = np.squeeze(test.astype('uint8'), axis = 3)
	# test = np.squeeze(test.astype('int8'), axis = 3)
	make_gif(result, f'output/predictions/{i}prediction.gif')
	make_gif(test, f'output/predictions/{i}locations.gif')
