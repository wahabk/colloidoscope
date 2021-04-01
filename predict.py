import matplotlib.pyplot as plt
import numpy as np
import time
import segmentation_models_3D as sm
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, schedules
timestr = time.strftime("%Y-%m-%d-%H-%M")
from make_dataset import *
from sim3d import *
import tensorflow as tf
from mainviewer import mainViewer
from unet import dataGenie
import trackpy as tp
import scipy

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
dataset = 'TF'
n_samples = 30
weightspath = 'output/Second.hdf5'
activation = 'sigmoid'



model = sm.Unet(BACKBONE, input_shape=input_shape, encoder_weights=None, classes=nclasses, activation=activation, encoder_freeze=False)

model.load_weights(weightspath)
test_list = []
position_list = []

# for n in range(1,300, 50):
# 	test = testGenie(n, dataset)
# 	test_list.append(test)
for n in range(1,10):
	test = testGenie(n, dataset)
	test_list.append(test)
test_list = np.array(test_list)
print(test_list.shape)


results = model.predict(test_list, batch_size=1) # read about this one


str_3D=np.array([[[0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]],

   [[0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]],

   [[0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]]], dtype='uint8')

for i, result in enumerate(results):
	result = results[i]*255
	test = test_list[i]*255
	print(result.shape, np.max(result), np.mean(result), np.min(result))
	result = np.squeeze(result.astype('uint8'), axis = 3)
	test = np.squeeze(test.astype('uint8'), axis = 3)
	resultLabel = result/255
	resultLabel[resultLabel<0.5] = 0
	resultLabel[resultLabel>0.5] = 1
	resultLabel = scipy.ndimage.label(resultLabel, structure=str_3D)
	# import pdb; pdb.set_trace()
	positions = scipy.ndimage.center_of_mass(result, resultLabel[0], index=range(1,resultLabel[1]))
	positions = np.array(positions)
	x, y = get_gr(positions, 50, 20)

	plt.plot(x, y)
	plt.savefig('output/gr.png')
	plt.cla()

	exit()

	# positions = tp.locate(result, 9, threshold=150)
	# positions = np.array([[positions.iloc[i]['z'], positions.iloc[i]['y'], positions.iloc[i]['x']] for i in range(1, len(positions))])
	# positions = [list(positions[:]['z']), list(positions[:]['y']), list(positions[:]['x'])]
	# print(type(positions), len(positions))
	# import pdb; pdb.set_trace()

	# mainViewer(test, labels=result)
	# test = np.squeeze(test.astype('int8'), axis = 3)
	# make_gif(result, f'output/predictions/{i}_label.gif', positions=positions, scale = 300)
	# make_gif(test, f'output/predictions/{i}_scan.gif', positions=positions, scale = 300)


