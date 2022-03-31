import os
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from data import cryodata
from models import deephemnma
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from models import loss
from utils import read_pdb
from torch.utils.tensorboard import SummaryWriter

def train(imgs, amplitudes, angles, shifts, epochs=400, batch_size=2, lr=1e-4, flag='all', mode='train', device='cuda'):
    num_epochs = epochs
    random_seed = 42
    validation_split = .2
    shuffle_dataset = True

    dataset = cryodata(imgs, amplitudes, angles, shifts, flag=flag, mode = mode,  transform=transforms.ToTensor())

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor((1-validation_split) * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    print('the train set size is: {} images'.format(len(train_sampler)))
    print('the validation set size is: {} images'.format(len(valid_sampler)))
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    if args.flag=='nma':
        model = deephemnma(4).to('cuda:0')

    elif args.flag=='ang':
        model = deephemnma(4).to('cuda:0')
    else:
        model = deephemnma(2).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #print(next(iter(train_loader)))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    criterion = nn.L1Loss()
    writer = SummaryWriter('./scalars')
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        running_loss = 0.0

        for img, params in train_loader:
            pred_params = model(img.to('cuda:0'), 'train')
            l = criterion(params.to('cuda:0'), pred_params)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
            epoch_loss += pred_params.shape[0] * l.item()
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, l.item() ), end='\r')
        valid_loss = 0.0
        with torch.no_grad():
            for img, params in validation_loader:
                pred_params = model(img.to('cuda:0'), 'validation')
                l = criterion(params.to('cuda:0'), pred_params)
                valid_loss += pred_params.shape[0] * l.item()

        print('epoch [{}/{}], train loss:{:.4f}, validation loss:{:.4f}'
              .format(epoch + 1, num_epochs, epoch_loss / len(train_loader.dataset), valid_loss / len(validation_loader.dataset)))
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader.dataset), epoch+1)
        writer.add_scalar('Loss/validation', valid_loss / len(validation_loader.dataset), epoch+1)
        scheduler.step(epoch_loss)
        torch.save(model.state_dict(), './resnet_based.pth')

if __name__ == '__main__':

    train(args)