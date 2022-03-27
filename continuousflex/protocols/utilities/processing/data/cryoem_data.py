import glob
from torch.utils.data import Dataset
from utils import spi2array, create_array
import torch
class cryodata(Dataset):

    def __init__(self, path, metadata_path, flag='nma', mode = 'train', transform=None):
        self.path = path
        self.metadata_path = metadata_path
        self.flag = flag
        self.files = sorted(glob.glob(self.path + "*.spi"))
        self.mode = mode
        if mode == 'train':
            self.amplitudes, img_names = create_array(self.metadata_path, 'nma')
            self.angles, img_names = create_array(self.metadata_path, 'ang')
            self.shifts, img_names = create_array(self.metadata_path, 'shf')
        else:
            pass
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if self.mode == 'train':
            if self.flag == 'nma':
                amplitudes = self.amplitudes[item]
                image_name = self.files[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    amplitudes = torch.tensor(amplitudes)
                return spi_array, amplitudes
        
            elif self.flag == 'ang':
                angles = self.angles[item]
                image_name = self.files[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    angles = torch.tensor(angles) 
                return spi_array, angles, image_name
            else:
                shifts = self.shifts[item]    
                image_name = self.files[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    shifts = torch.tensor(shifts)
                return spi_array, shifts
        else:
            image_name = self.files[item]
            spi_array = spi2array(image_name)
            if self.transform:
                spi_array = self.transform(spi_array)
            print(image_name)
            return spi_array, image_name
