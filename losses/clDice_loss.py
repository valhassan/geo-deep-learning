import torch
import torch.nn as nn
import torch.nn.functional as F


""" 
Source: 
https://github.com/jocpae/clDice
https://github.com/qubvel/segmentation_models.pytorch

"""
def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel

def mode_func(output: torch.Tensor, target: torch.Tensor, mode: str, ignore_index=None):
    bs = target.size(0)
    num_classes = output.size(1)

    if mode == 'binary':
        output = output.view(bs, 1, -1)
        target = target.view(bs, 1, -1)

        if ignore_index is not None:
            mask = target != ignore_index
            output = output * mask
            target = target * mask
    if mode == 'multiclass':
        target = target.view(bs, -1)
        output = output.view(bs, num_classes, -1)
        if ignore_index is not None:
            mask = target != ignore_index
            output = output * mask.unsqueeze(1)

            target = F.one_hot((target * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
            target = target.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
        else:
            target = F.one_hot(target, num_classes)  # N,H*W -> N,H*W, C
            target = target.permute(0, 2, 1)  # N, C, H*W
    assert output.size() == target.size()
    return output, target

def soft_dice_score(output: torch.Tensor,
                    target: torch.Tensor,
                    smooth: float,
                    mode: str,
                    ignore_index=None,) -> torch.Tensor:
    eps = 1e-7
    dims = (0, 2)
    output, target = mode_func(output, target, mode, ignore_index)
    intersection = torch.sum(output * target, dim=dims)
    cardinality = torch.sum(output + target, dim=dims)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    # print('dice_score', dice_score)
    loss = 1. - dice_score
    mask = target.sum(dims) > 0
    loss *= mask.to(loss.dtype)
    return loss.mean()

def soft_cldice_score(skel_output: torch.Tensor,
                      skel_target: torch.Tensor,
                      smooth: float,
                      mode: str,
                      ignore_index=None,) -> torch.Tensor:
    dims = (0, 2)
    skel_output, skel_target = mode_func(skel_output, skel_target, mode, ignore_index)
    tprec = (torch.sum(skel_output * skel_target, dim=dims) + smooth) / (torch.sum(skel_output, dim=dims) + smooth)
    tsens = (torch.sum(skel_target * skel_output, dim=dims) + smooth) / (torch.sum(skel_target, dim=dims) + smooth)
    cl_dice_score = 2.0 * (tprec * tsens) / (tprec + tsens)
    loss = 1 - cl_dice_score
    mask = skel_target.sum(dims) > 0
    loss *= mask.to(loss.dtype)
    return loss.mean()



class ClDice(nn.Module):

    def __init__(self, iter_: int=10, alpha: float=0.5, smooth: float=1., ignore_index=None, mode='multiclass'):
        super().__init__()
        self.mode = mode
        self.iter = iter_
        self.alpha = alpha
        self.smooth = smooth
        self.ignore_index = ignore_index
        assert mode in {'binary', 'multiclass'}

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        if self.mode == 'multiclass':
            output = output.log_softmax(dim=1).exp()
        else:
            output = F.logsigmoid(output).exp()
        dice = soft_dice_score(output, target, self.smooth, self.mode, self.ignore_index)
        skel_output = soft_skel(output, self.iter)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        skel_target = soft_skel(target.type_as(output), self.iter)
        cl_dice = soft_cldice_score(skel_output, skel_target, self.smooth, self.mode, self.ignore_index)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice









