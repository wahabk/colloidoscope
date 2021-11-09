import torch 
import os
import sys
print(sys.version)
# from src.deepcolloid import DeepColloid
print(os.cpu_count())
print(torch.__version__)
print('Testing deepcolloid, torch cuda status: ', torch.cuda.is_available())