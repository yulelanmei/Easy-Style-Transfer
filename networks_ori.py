import torch as th 
import torch.nn as nn 
import torch.nn.functional as F

class AdaIn(nn.Module):
    def __init__(self, eps= 1e-8):
        super(AdaIn, self).__init__()
        self.eps = eps
    def forward(self, c, s): 
        x = (c - th.mean(c, dim= 1, keepdim= True)) / (th.std(c, dim= 1, keepdim= True) + self.eps)
        x = (x * th.std(s, dim= 1, keepdim= True)) + th.mean(s, dim= 1, keepdim= True)
        return x
    
class SA_Block(nn.Module):
    def __init__(self, channels):
        super(SA_Block, self).__init__()
        self.c_trans = nn.Conv2d(channels, channels, (1, 1))
        self.s_trans = nn.Conv2d(channels, channels, (1, 1))
        self.i_trans = nn.Conv2d(channels, channels, (1, 1))
        self.softmax = nn.Softmax(dim= -1)
        self.o_trans = nn.Conv2d(channels, channels, (1, 1))

    def forward(self, c, s):
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
        of = of.view(c.size())
        of = self.o_trans(of)
        of = of + c
        return of

    def normalizer(self, tensor, eps= 1e-5): 
        size = tensor.size()
        b, c = size[:2]
        tensor_var = tensor.view(b, c, -1).var(dim= 2) + eps
        tensor_std = tensor_var.sqrt().view(b, c, 1, 1)
        tensor_mean = tensor.view(b, c, -1).mean(dim= 2).view(b, c, 1, 1)
        normalized_tensor = (tensor - tensor_mean.expand(size)) / tensor_std.expand(size)
        return normalized_tensor
    
def get_style_block(block_name: str):
    if block_name == 'adain': return AdaIn()
    if block_name == 'sablk': return SA_Block()
    
if __name__ == '__main__':
    
    device = 'cuda'
    
    ada = AdaIn().to(device)
    sa = SA_Block(3).to(device)
    
    c = th.rand((1, 3, 256, 256)).to(device)
    s = th.rand((1, 3, 256, 256)).to(device)
    
    result1 = ada(c, s)
    # result2 = sa(c, s)
    
    print(result1.size())
    # print(result2.size())
    