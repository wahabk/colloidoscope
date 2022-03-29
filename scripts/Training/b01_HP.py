import torch
import numpy as np
from colloidoscope.trainer import train
from colloidoscope.deepcolloid import DeepColloid
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
from ray import tune
import random
from ray.tune.schedulers import ASHAScheduler
from functools import partial

print(os.cpu_count())
print(torch.cuda.is_available())
print ('Current cuda device ', torch.cuda.current_device())
print('------------num available devices:', torch.cuda.device_count())

import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
	from https://github.com/Hsuxu/Loss_ToolBox-PyTorch
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss



if __name__ == "__main__":

	dataset_path = '/home/ak18001/Data/HDD/Colloids'
	# dataset_path = '/mnt/storage/home/ak18001/scratch/Colloids'
	# dataset_path = '/data/mb16907/wahab/Colloids'
	# dataset_path = '/user/home/ak18001/scratch/Colloids/' #bc4
	# dataset_path = '/user/home/ak18001/scratch/ak18001/Colloids' #bp1
	dc = DeepColloid(dataset_path)


	dataset_name = 'r_width'
	n_samples = dc.get_hdf5_keys(dataset_name)
	print(len(n_samples))
	all_data = list(range(1,3000))
	random.shuffle(all_data)

	num_samples = 10
	max_num_epochs = 50
	gpus_per_trial = 1

	train_data = all_data[0:600]
	val_data = all_data[601:800]
	test_data =	all_data[801:900]
	name = 'search focal'
	save = '/home/ak18001/code/colloidoscope/output/weights/unet.pt'
	device_ids = [0,]
	# save = '/user/home/ak18001/scratch/Colloids/unet.pt'
	
	config = {
		"lr": tune.choice([0.0001, 0.001]),
		"epochs": 10,
		"batch_size": tune.choice([4]),
		"n_blocks": tune.choice([2]),
		"norm": tune.choice(['batch']),
		"activation": tune.grid_search(['relu']),
		"start_filters": tune.choice([32]),
		"loss_function": tune.grid_search([BinaryFocalLoss(alpha=0, gamma=0), 
											BinaryFocalLoss(alpha=1, gamma=0), 
											BinaryFocalLoss(alpha=0, gamma=1), 
											BinaryFocalLoss(alpha=1, gamma=1)]),
	}

	# the scheduler will terminate badly performing trials
	scheduler = ASHAScheduler(
		metric="val_loss",
		mode="min",
		max_t=max_num_epochs,
		grace_period=1,
		reduction_factor=2)

	result = tune.run(
		partial(train, name=name, dataset_path=dataset_path, dataset_name=dataset_name, 
				train_data=train_data, val_data=val_data, test_data=test_data, save=save, 
				tuner=True, device_ids=device_ids),
		resources_per_trial={"cpu": 10, "gpu": 1},
		config=config,
		num_samples=num_samples,
		scheduler=None,
		checkpoint_at_end=True)
