from ast import arg
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F

class AdaIn(nn.Module):
    def __init__(self, eps= 1e-8):
        super(AdaIn, self).__init__()
        self.eps = eps
    def forward(self, c, s, x): 
        x = (x - th.mean(c, dim= 1, keepdim= True)) / (th.std(c, dim= 1, keepdim= True) + self.eps)
        x = (x * th.std(s, dim= 1, keepdim= True)) + th.mean(s, dim= 1, keepdim= True)
        return x 

class WCT(nn.Module): 
    def __init__(self, alpha= 0.7):
        super(WCT, self).__init__()
        self.alpha = alpha

    def get_wct_feature(self, cF, sF): 
        cFSize = cF.size()
        c_mean = th.mean(cF, 1) # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF) 
        cF = cF - c_mean

        contentConv = th.mm(cF, cF.t()).div(cFSize[1] - 1) + th.eye(cFSize[0]).double() # 
        _, c_e, c_v = th.svd(contentConv, some= False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = th.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)

        styleConv = th.mm(sF, sF.t()).div(sFSize[1] - 1)
        _, s_e, s_v = th.svd(styleConv, some= False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = th.mm(c_v[:, 0:k_c], th.diag(c_d))
        step2 = th.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = th.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = th.mm(th.mm(th.mm(s_v[:, 0:k_s], th.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def forward(self, c, s, x): 
        c = c.double()
        s = s.double()
        cb, cc, _, _ = c.size()
        sb, sc, _, _ = s.size()
        cF = c.view(cb, cc, -1) 
        sF = s.view(sb, sc, -1)
        Size = x.size() 

        cF = cF.squeeze_()
        sF = sF.squeeze_()
        tF = self.get_wct_feature(cF, sF)
        tF = tF.view(Size[1:])
        tF = tF.unsqueeze_(0)

        x = x.double()
        x = self.alpha * tF + (1 - self.alpha) * x
        x = x.float()
        return x

class get_FcMatrix(nn.Module): 
    def __init__(self, channels):
        super(get_FcMatrix, self).__init__()
        c = channels * channels
        self.fc = nn.Linear(c, c)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        x = th.bmm(x, x.transpose(1, 2)).div(h * w)
        x = x.view(b, -1)
        x = self.fc(x)
        x = x.view(b, c, c)
        return x

class LinearMatrix(nn.Module): 
    def __init__(self, channels):
        super(LinearMatrix, self).__init__()  
        self.c_trans = get_FcMatrix(channels)
        self.s_trans = get_FcMatrix(channels)

    def forward(self, c, s, x):     
        cb, cc, _, _ = c.size()
        c = c - th.mean(c.view(cb, cc, -1), dim= 2, keepdim= True).unsqueeze(3).expand_as(c)

        sb, sc, _, _ = s.size()
        sMean = th.mean(s.view(sb, sc, -1), dim= 2, keepdim= True).unsqueeze(3)
        sMeanC = sMean.expand_as(c)
        sMeanS = sMean.expand_as(s)
        s = s - sMeanS

        cM = self.c_trans(c)
        sM = self.s_trans(s)
        tM = th.bmm(sM, cM)

        xb, xc, xh, xw = x.size()
        x = x.view(xb, xc, -1)
        x = th.bmm(tM, x).view(xb, xc, xh, xw)
        x = x + sMeanC

        return x

class SA_Block(nn.Module):
    def __init__(self, channels):
        super(SA_Block, self).__init__()
        self.c_trans = nn.Conv2d(channels, channels, (1, 1))
        self.s_trans = nn.Conv2d(channels, channels, (1, 1))
        self.i_trans = nn.Conv2d(channels, channels, (1, 1))
        self.softmax = nn.Softmax(dim= -1)
        self.o_trans = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, c, s, x):
        cf = self.c_trans(self.normalizer(c))
        sf = self.s_trans(self.normalizer(s))
        xf = self.i_trans(s)

        b, _, h, w = cf.size()
        cf = cf.view(b, -1, h * w)
        b, _, h, w = sf.size()
        sf = sf.view(b, -1, h * w)
        b, _, h, w = xf.size()
        xf = xf.view(b, -1, h * w)

        mask = th.bmm(cf.permute(0, 2, 1), sf)
        mask = self.softmax(mask)

        of = th.bmm(xf, mask.permute(0, 2, 1))
        of = of.view(x.size())
        of = self.o_trans(of)
        x = x + of 
        return x

    def normalizer(self, tensor, eps= 1e-5): 
        size = tensor.size()
        b, c = size[:2]
        tensor_var = tensor.view(b, c, -1).var(dim= 2) + eps
        tensor_std = tensor_var.sqrt().view(b, c, 1, 1)
        tensor_mean = tensor.view(b, c, -1).mean(dim= 2).view(b, c, 1, 1)
        normalized_tensor = (tensor - tensor_mean.expand(size)) / tensor_std.expand(size)
        return normalized_tensor

def get_style_trans(method_name: str, arg_2nd= None):
    lower_method_name = method_name.lower()
    if lower_method_name == 'adain':
        if arg_2nd is not None:
            assert isinstance(arg_2nd, float)
            return AdaIn(arg_2nd)
        else: 
            return AdaIn()
    elif lower_method_name == 'wct':
        if arg_2nd is not None:
            assert isinstance(arg_2nd, float)
            return WCT(arg_2nd)
        else: 
            return WCT()
    elif lower_method_name == 'linearmatrix':
        if arg_2nd is not None: 
            assert isinstance(arg_2nd, int)
            return LinearMatrix(arg_2nd)
        else: 
            raise RuntimeError('There must be a channel arg!')
    elif lower_method_name == 'sablock':
        if arg_2nd is not None: 
            assert isinstance(arg_2nd, int)
            return SA_Block(arg_2nd)
        else: 
            raise RuntimeError('There must be a channel arg!')
    else: 
        raise RuntimeError('There is no corresponding style transfer!')

if __name__ == '__main__': 
    pass