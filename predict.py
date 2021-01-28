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
	scan, positions = read_hdf5(dataset, n, positions=True)
	scan = scan/255
	scan = scan[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
	return scan, positions


BACKBONE = 'resnet34'
val_steps = 19
batch_size = 1
steps_per_epoch = 80
epochs = 40
nclasses = 1
lr = 1e-5
input_shape = [32,128,128,1]
dataset = 'First'
n_samples = 30
weightspath = 'output/unet_checkpoints2.hdf5'
activation = 'sigmoid'



model = sm.Unet(BACKBONE, input_shape=input_shape, encoder_weights=None, classes=nclasses, activation=activation, encoder_freeze=False)

model.load_weights(weightspath)
test_list = []
position_list = []

for n in range(1, 5):
	test, positions = testGenie(1, dataset)
	test_list.append(test)
	position_list.append(positions)

test_list = np.array(test_list)
print(test_list.shape)


results = model.predict(test_list, batch_size=1) # read about this one
result = results[0]*255
test = test_list[0]*255
print(result.shape, np.max(result), np.mean(result), np.min(result))
result = np.squeeze(result.astype('uint8'), axis = 3)
test = np.squeeze(test.astype('uint8'), axis = 3)
# test = np.squeeze(test.astype('int8'), axis = 3)
make_gif(result, 'prediction.gif', positions=position_list[0])
make_gif(test, 'locations.gif', positions=position_list[0])
