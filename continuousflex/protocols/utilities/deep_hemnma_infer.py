import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from continuousflex.protocols.utilities.processing_dh.data import cryodata
from continuousflex.protocols.utilities.processing_dh.models import deephemnma
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import sys

def infer(imgs_path, weights_path, batch_size=2, flag=0, device=0, mode='inference'):
    FLAG = ''
    if flag==0:
        FLAG = 'nma'
    elif flag==1:
        FLAG = 'ang'
    elif flag==2:
        FLAG = 'shf'
    else:
        FLAG = 'all'
    DEVICE = ''
    if device==0:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'


    dataset = cryodata(imgs_path, flag=FLAG, mode = mode, transform=transforms.ToTensor())

    dataset_size = len(dataset)
    print('the train set size is: {} images'.format(dataset_size))

    data_loader = DataLoader(dataset, batch_size=batch_size)

    if FLAG=='nma':
        model = deephemnma(3).to(DEVICE)
        predictions = np.zeros((dataset_size, 3), dtype='float32')
    elif FLAG=='ang':
        model = deephemnma(4).to(DEVICE)
        predictions = np.zeros((dataset_size, 4), dtype='float32')
    elif FLAG=='shf':
        model = deephemnma(2).to(DEVICE)
        predictions = np.zeros((dataset_size, 2), dtype='float32')
    elif FLAG=='all':
        model = deephemnma(9).to(DEVICE)
        predictions = np.zeros((dataset_size, 9), dtype='float32')

    model.load_state_dict(torch.load(weights_path))
    with torch.no_grad():
        i = 0
        for img, params in data_loader:
            pred_params = model(img.to(DEVICE), mode)
            predictions[i * batch_size:(i + 1) * batch_size, :] = pred_params.detach()
            i+=1

if __name__ == '__main__':
    infer(sys.argv[0],
          sys.argv[1],
          int(sys.argv[2]),
          int(sys.argv[3]),
          int(sys.argv[4]),
          sys.argv[5])
    sys.exit()