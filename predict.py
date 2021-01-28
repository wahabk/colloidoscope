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

def testGenie(n):
	scan, positions = read_hdf5('Test', n, positions=True)
	scan = scan/255
	scan = scan[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
	return scan, positions


dataset = 'Test'
input_shape = [32,256,256,1]
modelpath = 'output/unet_checkpoints2.hdf5'
BACKBONE = 'resnet34'

datagenie = dataGenie(batch_size=1,
                        data_gen_args=dict(),
                        dataset=dataset,
                        n_samples=range(1, 3))

model = sm.Unet(BACKBONE, encoder_weights=None, input_shape=input_shape, classes=1, activation='sigmoid')
model.load_weights(modelpath)
test_list = []
position_list = []

for n in range(1, 5):
	test, positions = testGenie(1)
	test_list.append(test)
	position_list.append(positions)

test_list = np.array(test_list)
print(test_list.shape)


results = model.predict(test_list, batch_size=1) # read about this one
result = results[0]*255
print(result.shape, np.max(result), np.mean(result), np.min(result))
result = np.squeeze(result.astype('uint8'), axis = 3)
# test = np.squeeze(test.astype('int8'), axis = 3)
mainViewer(result, positions=position_list[0])

