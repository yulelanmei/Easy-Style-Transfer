import torch as th 
import torch.nn as nn 

class GramMatrix(nn.Module): 
    def forward(self, input, is_norm= True):
        b, c, h, w = input.size()
        tensor = input.view(b, c, h * w)
        gram = th.bmm(tensor, tensor.transpose(1, 2))  
        if is_norm: 
            gram.div_(c * h * w) 
        return gram

class Decoder_Train_Loss(nn.Module):
    def __init__(self):
        super(Decoder_Train_Loss, self).__init__()
        from mssim import MSSSIM
        self.l2Loss = nn.MSELoss()
        self.mssim  = MSSSIM()

    def forward(self, dlist: list, elist: list, weighted= False):
        loss = self.img_similarity(dlist[0], elist[0], weighted)
        for i in range(1, 5): 
            loss = loss + self.tensor_similarity(dlist[i], elist[i])
        return loss
    
    def img_similarity(self, input, target, weighted): 
        loss = self.l2Loss(input, target)
        if weighted: 
            weight = self.mssim(input, target)
            loss = loss / (weight + 1e-8)
        return loss 

    def tensor_similarity(self, input, target): 
        loss = self.l2Loss(input, target)
        return loss 

class Gram_Style_Loss(nn.Module): 
    def __init__(self):
        super(Gram_Style_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.get_gram = GramMatrix()

    def forward(self, input, target): 
        _, ic, _, _ = input.size()
        loss = self.mse_loss(self.get_gram(input), self.get_gram(target))
        return loss / (2 * ic ** 2) 

class Content_Loss(nn.Module): 
    def __init__(self):
        super(Content_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target): 
        ib, ic, _, _ = input.size()
        tensor_i = input.view(ib, ic, -1) 

        tb, tc, _, _ = target.size()
        tensor_t = target.view(tb, tc, -1)

        iMean = th.mean(tensor_i, dim= 2)
        tMean = th.mean(tensor_t, dim= 2)

        loss = self.mse_loss(iMean, tMean)
        return loss

class TV_Loss(nn.Module):
    def __init__(self, weight= 1):
        super(TV_Loss, self).__init__()
        self.weight = weight

    def forward(self, x):
        b, _, h, w = x.size()
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = th.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv = th.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    def _tensor_size(self, t):
        _, c, h, w = t.size()
        return c * h * w

class Aware_Style_Loss(nn.Module):
    def __init__(self): 
        super(Aware_Style_Loss, self).__init__()
        self.gram = GramMatrix()

    def forward(self, input, target): 
        g1 = self.gram(input, False)
        g2 = self.gram(target, False).detach()

        size = input.size()
        assert(len(size) == 4)

        g1_norm = th.linalg.norm(g1, dim = (1, 2))
        g2_norm = th.linalg.norm(g2, dim = (1, 2))

        size = g1.size()
        Nl = size[1] * size[2] # Or C x C = C^2
        normalize_term =  (th.square(g1_norm) + th.square(g2_norm)) / Nl 

        weights = (1 / normalize_term)
        #weights = weights.view(size[0],1,1)
        return self.weighted_mse_loss(g1, g2, weights)

    def weighted_mse_loss(self, input, target, weights = None):
        assert input.size() == target.size()
        size = input.size()
        if weights == None:
            weights = th.ones(size = size[0])
    
        if len(size) == 3: # gram matrix is B,C,C
            se = ((input.view(size[0], -1) - target.view(size[0], -1)) ** 2)
        return (se.mean(dim = 1) * weights).mean()

def get_target_loss_fn(loss_name: str):
    lower_loss_name = loss_name.lower()
    if lower_loss_name == 'gram_style_loss': 
        return Gram_Style_Loss()
    elif lower_loss_name == 'content_loss': 
        return Content_Loss()
    elif lower_loss_name == 'aware_style_loss': 
        return Aware_Style_Loss()
    else:
        raise RuntimeError('There is no corresponding loss function!')

def get_self_loss_fn(loss_name: str): 
    lower_loss_name = loss_name.lower()
    if lower_loss_name == 'tv_loss': 
        return TV_Loss()
    else: 
        raise RuntimeError('There is no corresponding loss function!')

class LossUnit(nn.Module):
    def __init__(self, c_cfg: list, s_cfg: list, c_w: list, s_w: list, y_cfg: list = None):
        super(LossUnit, self).__init__()
        assert len(c_cfg) == len(c_w) and len(s_cfg) == len(s_w)
        self.cLossList = nn.ModuleList([get_target_loss_fn(loss_name) for loss_name in c_cfg])
        self.sLossList = nn.ModuleList([get_target_loss_fn(loss_name) for loss_name in s_cfg])
        assert self.cLossList.__len__() > 0 and self.sLossList.__len__() > 0
        self.cLossWeights = c_w
        self.sLossWeights = s_w
        self.if_y_loss = False
        if y_cfg is not None: 
            self.if_y_loss = True
            self.yLossList = nn.ModuleList([get_self_loss_fn(loss_name) for loss_name in y_cfg])
            assert self.yLossList.__len__() > 0

    def forward(self, c, s, x): 
        loss = self.calc_target_loss(x, c, self.cLossList, self.cLossWeights) \
             + self.calc_target_loss(x, s, self.sLossList, self.sLossWeights)
        if self.if_y_loss: 
            loss = loss + self.calc_self_loss(x, self.yLossList)
        return 

    def calc_target_loss(self, input, target, lossList: nn.ModuleList, lossWeights: list):
        loss = lossWeights[0] * lossList[0](input, target)
        for i in range(1, lossList.__len__()): 
            loss = loss + lossWeights[i] * lossList[i](input, target)
        return loss 

    def calc_self_loss(self, input, lossList: nn.ModuleList): 
        loss = lossList[0](input)
        for i in range(1, lossList.__len__()): 
            loss = loss + lossList[i](input)
        return loss 
    
class StyleTransferLoss(nn.Module):
    def __init__(self, loss_cfg: list): 
        super(StyleTransferLoss, self).__init__()
        self.lossList = nn.ModuleList([
            
        ])

    def forward(self, dlist: list, eclist: list, eslist: list): 
        
        return  