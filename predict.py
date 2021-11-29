import torch
import numpy as np
from src.trainer import Trainer, plot_training, LearningRateFinder
from src.unet import UNet
import torchio as tio
from src.dataset import ColloidsDatasetSimulated
from src.deepcolloid import DeepColloid
import matplotlib.pyplot as plt
import neptune.new as neptune

def predict(array, threshold, model, weights_path, device, return_positions=False):
    
    model_weights = torch.load(weights_path)
    model.load_state_dict(model_weights)

    x = torch.from_numpy(array).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_sigmoid = torch.sigmoid(out, dim=3)  # perform softmax on output
    label = out_sigmoid > threshold
    
    # post process to numpy array
    label = label.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    label = np.squeeze(label)  # remove batch dim and channel dim -> [H, W]

    return label


dataset_path = '/home/ak18001/Data/HDD/Colloids'
# dataset_path = '/home/wahab/Data/HDD/Colloids'
# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
dc = DeepColloid(dataset_path)
# dc = DeepColloid(dataset_path)


roiSize = (32,128,128)
batch_size = 2
num_workers = 2


# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'predict on {device}')

# model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


weights_path =  'output/weights/unet.pt'



predict(array, threshold, model, weights_path, device, return_positions=False)


