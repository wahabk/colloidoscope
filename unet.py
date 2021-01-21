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
from generator import customImageDataGenerator

def lr_scheduler(epoch, learning_rate):
    decay_rate = 0.1
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
        
        print('[dataGenie] Initialising image and mask generators')
        image_generator = imagegen.flow(scan_list,
            batch_size = batch_size,
            #save_to_dir = 'output/Keras/',
            save_prefix = 'dataGenie',
            seed = 1,
            shuffle=False
            )
        mask_generator = maskgen.flow(label_list, 
            batch_size = batch_size,
            #save_to_dir = 'output/Keras/',
            save_prefix = 'dataGenie',
            seed = 1,
            shuffle=False
            )
        print('Ready.')

        datagen = zip(image_generator, mask_generator)
        for x_batch, y_batch in datagen:
            x_batch = x_batch/255
            y_batch = y_batch/255
            # print(x_batch[0].shape, x_batch[0].dtype, np.amax(x_batch[0]))
            # print(y_batch[0].shape, y_batch[0].dtype, np.amax(y_batch[0]))
            yield (x_batch, y_batch)

def testGenie(n):
	scan, positions = read_hdf5('Test', n, positions=True)
	scan = scan/255
	scan = scan[:,:,:,np.newaxis] # add final axis to show datagens its grayscale
	return scan, positions
if __name__ == "__main__":

    data_gen_args = dict(rotation_range=0.05,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        vertical_flip = True,
                        fill_mode='constant',
                        cval = 0)

    BACKBONE = 'resnet34'
    val_steps = 10
    batch_size = 1
    steps_per_epoch = 20
    epochs = 1
    lr = 1e-4
    input_shape = [32,256,256,1]
    dataset = 'Test'
    n_samples = 30
    modelpath = 'output/unet_checkpoints2.hdf5'

    model_checkpoint = ModelCheckpoint(modelpath, 
                                            monitor = 'loss', verbose = 1, save_best_only = True)

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
        model_checkpoint
    ]

    datagenie = dataGenie(batch_size=batch_size,
                            data_gen_args=data_gen_args,
                            dataset=dataset,
                            n_samples=range(1, 21))

    valdatagenie = dataGenie(batch_size=batch_size,
                            data_gen_args=data_gen_args,
                            dataset=dataset,
                            n_samples=range(21,31))



    opt = Adam(lr=lr)
    model = sm.Unet(BACKBONE, encoder_weights=None, input_shape=input_shape, classes=1, activation='sigmoid')

    # model.load_weights(modelpath)
    model.compile(optimizer='adam', loss=sm.losses.DiceLoss(), metrics=[sm.losses.DiceLoss()])
    history = model.fit(datagenie, validation_data=valdatagenie, steps_per_epoch = steps_per_epoch, 
                        epochs = epochs, callbacks=callbacks, validation_steps=val_steps)

    test, positions = testGenie(1)
    print(test.shape)
    results = model.predict(test, batch_size=1) # read about this one
    print(results.shapes)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Unet loss (lr={lr})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0,1)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'output/loss_curves/{timestr}_loss.png')
