import torch
from colloidoscope.dataset import ColloidsDatasetSimulated


check_ds = ColloidsDatasetSimulated(dataset_path, params['dataset_name'], params['train_data'], transform=train_segtrans, label_transform=train_segtrans) 
check_loader = torch.utils.data.DataLoader(check_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=torch.cuda.is_available())
x, y = example = next(iter(check_loader))[0]
print(x.shape, x.max(), x.min())
print(y.shape, y.max(), y.min())