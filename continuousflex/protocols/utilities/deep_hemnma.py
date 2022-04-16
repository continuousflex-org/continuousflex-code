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

def train(imgs_path, output_path, epochs=400, batch_size=2, lr=1e-4, flag=0, device=0, mode='train'):

    num_epochs = epochs
    random_seed = 42
    validation_split = .2
    shuffle_dataset = True
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


    dataset = cryodata(imgs_path, output_path, flag=FLAG, mode = mode, transform=transforms.ToTensor())
    print("****************************************************")
    print(output_path)
    print("****************************************************")
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
    im, p = next(iter(train_loader))
    if FLAG=='nma':
        model = deephemnma(p.shape[1]).to(DEVICE)

    elif FLAG=='ang':
        model = deephemnma(p.shape[1]).to(DEVICE)
    elif FLAG=='shf':
        model = deephemnma(p.shape[1]).to(DEVICE)
    elif FLAG=='all':
        model = deephemnma(p.shape[1]).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    criterion = nn.L1Loss()
    writer = SummaryWriter(output_path+'/scalars')
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        running_loss = 0.0

        for img, params in train_loader:
            pred_params = model(img.to(DEVICE), 'train')
            l = criterion(params.to(DEVICE), pred_params)
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
                pred_params = model(img.to(DEVICE), 'validation')
                l = criterion(params.to(DEVICE), pred_params)
                valid_loss += pred_params.shape[0] * l.item()

        print('epoch [{}/{}], train loss:{:.4f}, validation loss:{:.4f}'
              .format(epoch + 1, num_epochs, epoch_loss / len(train_loader.dataset), valid_loss / len(validation_loader.dataset)))
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader.dataset), epoch+1)
        writer.add_scalar('Loss/validation', valid_loss / len(validation_loader.dataset), epoch+1)
        scheduler.step(epoch_loss)
        torch.save(model.state_dict(), output_path+'/weights.pth')

if __name__ == '__main__':
    train(sys.argv[1],
          sys.argv[2],
          int(sys.argv[3]),
          int(sys.argv[4]),
          float(sys.argv[5]),
          int(sys.argv[6]),
          int(sys.argv[7]))