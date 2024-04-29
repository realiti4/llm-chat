import os
import torch

from llama import Dialog, Llama


os.environ['LOCAL_RANK'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

