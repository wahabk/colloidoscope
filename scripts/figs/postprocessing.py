from sklearn.inspection import plot_partial_dependence
from colloidoscope import DeepColloid
from colloidoscope.predict import find_positions
from colloidoscope.train_utils import plot_gr
from colloidoscope.hoomd_sim_positions import read_gsd, convert_hoomd_positions
# from colloidoscope.simulator import crop_positions_for_label
import numpy as np
import matplotlib.pyplot as plt
import napari
from random import randrange, uniform, triangular
import numpy as np
import random
from scipy import ndimage
import math
from scipy.signal import convolve2d
from pathlib2 import Path
from numba import njit
import torch

from scipy.spatial.distance import pdist, cdist

from tqdm import tqdm


from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max, blob_dog, blob_log

from math import sqrt






if __name__ == '__main__':
    dataset_path = '/mnt/scratch/ak18001/Colloids/'
    # dataset_path = '/home/ak18001/Data/HDD/Colloids'
    # dataset_path = '/home/wahab/Data/HDD/Colloids'
    # dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
    dc = DeepColloid(dataset_path)

    canvas_size=(100,100,100)
    label_size=(96,96,96)
    dataset_name = 'heatmap_3000_2_test'


    index = 10000
    r = 5
    d = r*2
    volfrac = 0.1
    threshold = 0.5

    # d = dc.read_hdf5(dataset_name, index)
    # image = d['image']
    # label = d["label"]
    # # positions = d["positions"]
    # # metadata = d["metadata"]
    # # diameters = d["diameters"]
    # # label[label<threshold] = 0

    # image = torch.tensor(image/image.max())
    # label = torch.tensor(label/label.max())

    # image = image.cpu().numpy()
    # label = label.cpu().numpy()

    # metadata, true_positions, diameters = dc.read_metadata(dataset_name, index)
    # print(metadata)

    # # label = dc.crop3d(label, label_size)

    # print(image.max(), image.min(), image.dtype, dc.round_up_to_odd(metadata['params']['r']*2))
    # tp_pred, df = dc.run_trackpy(image, diameter = dc.round_up_to_odd(metadata['params']['r']*2), )

    # positions, diameters = crop_positions_for_label(positions, canvas_size, label_size, diameters)

    # print(label.dtype)
    # print(label.shape, label.max(), label.min())

    # distance = ndi.distance_transform_edt(np.array(label*255, dtype="uint16"))
    # coords = peak_local_max(label)
    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(coords.T)] = True
    # markers, _ = ndi.label(mask)
    # new_label = watershed(-distance, markers, mask=label)

    # print(distance.shape, distance.max(), distance.min())
    # print(coords.shape, coords.max(), coords.min())
    # print(np.shape(coords))
    # print(mask.shape, mask.max(), mask.min())
    # print(new_label.shape, new_label.max(), new_label.min())

    # import pdb; pdb.set_trace()

    # coords = np.array(coords)
    # coords = blob_log(label, min_sigma=diameter, max_sigma=diameter, overlap=0)
    # label[label<0.5] = 0
    # coords = peak_local_max(label, min_distance=metadata['params']['r'])
    # sigma = int((metadata['params']['r'])/sqrt(3))
    # # coords = blob_log(label, min_sigma=sigma, max_sigma=sigma, overlap=0)[:,:-1] # get rid of sigmas
    # coords = find_positions(label, threshold=0)
    # # print(len(coords))
    # # print(coords)
    # prec, rec = dc.get_precision_recall(true_positions, tp_pred, diameters=diameters, threshold=0.5)
    # print('tp', prec, rec)
    # prec, rec = dc.get_precision_recall(true_positions, coords, diameters=diameters, threshold=0.5)
    # print('local_max', prec, rec)

    # ap, precisions, recalls, thresholds = dc.average_precision(true_positions, coords, diameters)

    # fig = dc.plot_pr(ap, precisions, recalls, thresholds, name='Unet', tag='o-', color='red')
    # # plt.show()

    # x,y = dc.get_gr(true_positions, 50, 50, )
    # plt.plot(x, y, label=f'true n ={len(true_positions)}', color='black')
    # x,y = dc.get_gr(tp_pred, 50, 50, )
    # plt.plot(x, y, label=f'trackpy n ={len(tp_pred)}', color='grey')
    # x,y = dc.get_gr(coords, 50, 50, )
    # plt.plot(x, y, label=f'local_max n ={len(coords)}', color='red')
    # plt.legend()
    # plt.savefig("postprocessing_grs.png")
    # plt.show()


    # dc.view(image, label=label, positions=coords)

    fig, axs = plt.subplots(2, 2)
    plt.tight_layout()

    ims = ["Image", "Prediction"]
    processing_methods = ["True", "TP", "MAX", "LOG"]



    d = dc.read_hdf5(dataset_name, index)
    image = d['image']
    label = d["label"]

    image = np.array(image/image.max(),  dtype="float32")
    label = np.array(label/label.max(),  dtype="float32")
    image = dc.crop3d(image, (100,100,100))
    label = dc.crop3d(label, (100,100,100))

    metadata, true_positions, diameters = dc.read_metadata(dataset_name, index)
    print(metadata)


    for i, (name, array) in enumerate(zip(ims, [image, label])):
        print(name)
        projection = np.array(np.max(array, axis=0)*255, dtype="uint8")
        axs[0,i].imshow(projection)
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[0,i].set_title(name)
        
        tp_pred, df = dc.run_trackpy(array, diameter = dc.round_up_to_odd(metadata['params']['r']*2), )
        
        max_pred = find_positions(array, threshold=0)
        
        
        ap, precisions, recalls, thresholds = dc.average_precision(true_positions, true_positions, diameters)
        dc.plot_pr(ap, precisions, recalls, thresholds, name='True', tag='o-', color='red', axs=axs[1,i])
        ap, precisions, recalls, thresholds = dc.average_precision(true_positions, tp_pred, diameters)
        dc.plot_pr(ap, precisions, recalls, thresholds, name='TP', tag='o-', color='red', axs=axs[1,i])
        ap, precisions, recalls, thresholds = dc.average_precision(true_positions, max_pred, diameters)
        dc.plot_pr(ap, precisions, recalls, thresholds, name='MAX', tag='o-', color='red', axs=axs[1,i])
        axs[1,i].legend()
        
        # sigma = int((metadata['params']['r'])/sqrt(3))
        # log_pred = blob_log(label, min_sigma=sigma, max_sigma=sigma, overlap=0)[:,:-1] # get rid of sigmas
        # x,y = dc.get_gr(true_positions, 50, 50, )
        # axs[1,i].plot(x, y, 'b-', label=f'true n ={len(true_positions)}')
        # x,y = dc.get_gr(tp_pred, 50, 50, )
        # axs[1,i].plot(x, y, label=f'trackpy n ={len(tp_pred)}', color='grey')
        # x,y = dc.get_gr(max_pred, 50, 50, )
        # axs[1,i].plot(x, y, label=f'local_max n ={len(max_pred)}', color='red')
        # # x,y = dc.get_gr(log_pred, 50, 50, )
        # # axs[1,i].plot(x, y, label=f'local_max n ={len(coords)}', color='green')
        # axs[1,i].legend()
        
        # x, y = dc.get_gr(trackpy_pos, 100, 100)
        # plot_gr(x, y, int_diameter, label=f'TP n={len(trackpy_pos)}', color='gray', axs=axs[i,3], fontsize=16)
        # x, y = dc.get_gr(pred_positions, 100, 100)
        # plot_gr(x, y, int_diameter, label=f'U-net n={len(pred_positions)}', color='red', axs=axs[i,3], fontsize=16)
		# 	if i != real_len-1:
		# 		axs[i,3].set_xlabel("")
		# 		axs[i,3].set_xticks([])

    fig.set_figheight(6)
    fig.set_figheight(6)
    plt.savefig("output/figs/postprocessing_grs.png", bbox_inches="tight")