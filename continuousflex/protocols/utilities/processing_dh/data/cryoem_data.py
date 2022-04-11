import glob
from torch.utils.data import Dataset
from continuousflex.protocols.utilities.processing_dh.utils import spi2array
import torch
import pwem.emlib.metadata as md
class cryodata(Dataset):

    def __init__(self, path, flag='nma', mode = 'train', transform=None):
        self.path = path
        self.flag = flag
        self.mode = mode
        mdImgs = md.MetaData(self.path)
        rot = []
        tilt = []
        psi = []
        nma = []
        shift_x = []
        shift_y = []
        imgPath = []
        for objId in mdImgs:
            imgPath.append(mdImgs.getValue(md.MDL_IMAGE, objId))
            rot.append(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt.append(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi.append(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shift_x.append(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shift_y.append(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            nma.append(mdImgs.getValue(md.MDL_NMA, objId))
        self.images_Path = imgPath
        rot_ = torch.tensor(rot)
        tilt_ = torch.tensor(tilt)
        psi_ = torch.tensor(psi)
        shiftx = torch.tensor(shift_x)
        shifty = torch.tensor(shift_y)
        if mode == 'train':
            self.angles = torch.column_stack((rot_, tilt_, psi_))
            self.shifts = torch.column_stack((shiftx, shifty))
            self.amplitudes = torch.tensor(nma, dtype=torch.float32)
        else:
            pass

        self.transform = transform

    def __len__(self):
        return len(self.images_Path)

    def __getitem__(self, item):
        if self.mode == 'train':
            if self.flag == 'nma':
                amplitudes = self.amplitudes[item]
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    amplitudes = torch.tensor(amplitudes)
                return spi_array, amplitudes
            elif self.flag == 'ang':
                angles = self.angles[item]
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    angles = torch.tensor(angles) 
                return spi_array, angles, image_name
            else:
                shifts = self.shifts[item]    
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    shifts = torch.tensor(shifts)
                return spi_array, shifts
        else:
            image_name = self.images_Path[item]
            spi_array = spi2array(image_name)
            if self.transform:
                spi_array = self.transform(spi_array)
            return spi_array, image_name