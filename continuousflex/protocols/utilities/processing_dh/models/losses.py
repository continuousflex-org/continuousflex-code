import torch
import torch.nn.functional as F
import numpy as np


def loss(*args):
    inp_img = args[0]
    pred_img = args[1]
    inp = args[2]
    pred = args[3]
    
    l1 = L1(inp, pred)
    cc_loss = cc(inp_img, pred_img)
    
    loss = l1+cc_loss
    return loss

def mse(*args):
    input_image = args[0]
    projected_image = args[1]
    loss = F.mse_loss(input_image, projected_image)
    return loss



def cc(img1, img2, reduction = 'mean'):
    img1 = img1 - torch.mean(img1, dim = (1, 2))[...,None, None]
    img2 = img2 - torch.mean(img2, dim = (1, 2))[...,None, None]
    cc = torch.sum(img1*img2, dim = (1,2))
    p1 = torch.sqrt(torch.sum(img1*img1, dim=(1,2)))
    p2 = torch.sqrt(torch.sum(img2*img2, dim=(1,2)))
    ncc = cc/(p1*p2)
    if reduction == 'mean':
        res = torch.mean(ncc)
        return res
    elif reduction == 'sum':
        res = torch.sum(ncc)
        return res
    else:
        raise ValueError('Unknown flag, you must select reduce or sum for loss redeuction mode')    
        
        
        
def L1(*args):
    input = args[0]
    infer = args[1]
    loss = F.l1_loss(input, infer)
    return loss

