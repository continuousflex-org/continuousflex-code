import torch.nn as nn
import torch
from . import ResNet, mlp, Bottleneck, BasicBlock
#from utils import projectPDB_NP, normalize, torch_normalize, quater2euler



class deephemnma(nn.Module):
    def __init__(self, output):
        super(deephemnma, self).__init__()
        self.output = output
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3])
        self.mlp = mlp(output)
    """
    def forward(self, x, pdb, mode = 'train'):
        resnet = self.resnet(x)
        flat = torch.flatten(resnet, start_dim=1)
        mlp = self.mlp(flat)
        if mode == 'train':
            proj_imgs = torch.ones_like(x)
            for i in range(mlp.shape[0]):
                mlp_ = quater2euler(mlp[i,:])
                proj_imgs[i] = torch_normalize(projectPDB_NP(pdb.to('cpu'), 128, 2.96, 1, mlp_[0].to('cpu'), mlp_[1].to('cpu'), mlp_[2].to('cpu'), 0, 0, 0))
            return mlp, proj_imgs.to('cuda:0')
        elif mode == 'inference':
            return mlp
    """
    def forward(self, x, mode = 'train'):
        resnet = self.resnet(x)
        flat = torch.flatten(resnet, start_dim=1)
        mlp = self.mlp(flat)
        return mlp
    
