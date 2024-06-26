import torch as th
import torch.nn as nn 
import torchvision.models as model
import networks as net
import networks_ori as styleblock
import config as cfg
from typing import Optional

# ---------------   VGG19   ----------------
def get_decode_block(in_channels, out_channels, num_of_layer, if_scaled = True): 
    block = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)), 
                          nn.Conv2d(in_channels, out_channels, (3, 3)),
                          nn.ReLU(inplace= True))
    count = 3
    if if_scaled: 
        block.add_module(str(count), nn.Upsample(scale_factor= 2)) 
        count += 1
    for _ in range(num_of_layer): 
        block.add_module(str(count), nn.ReflectionPad2d((1, 1, 1, 1)))
        count += 1
        block.add_module(str(count), nn.Conv2d(out_channels, out_channels, (3, 3)))
        count += 1
        block.add_module(str(count), nn.ReLU(inplace= True))
        count += 1
    return block

def get_vgg19_decoder(decoder_config: tuple, num_of_layers): 
    assert num_of_layers >= 1 and num_of_layers <= 5
    return nn.ModuleList([get_decode_block(*(decoder_config[i])) for i in range(5 - num_of_layers, 5)])

def get_vgg19(vgg_config: tuple, num_of_layers, vgg):
    assert num_of_layers >= 1 and num_of_layers <= 5
    return nn.ModuleList([
        nn.Sequential(*list(vgg.children())[vgg_config[i - 1] : vgg_config[i]]) for i in range(1, num_of_layers + 1)
    ]).requires_grad_(False)
    
def get_InstanceNorm(norm_config: tuple, num_of_layers):
    assert num_of_layers >= 1 and num_of_layers <= 5
    return nn.ModuleList([nn.InstanceNorm2d(norm_config[i][1]) for i in range(num_of_layers)])

class Decoder_Train_Net(nn.Module): 
    def __init__(self, get_layer_output= False, init= False, Normalized= False):
        super(Decoder_Train_Net, self).__init__()
        vgg = model.vgg19()
        vgg.load_state_dict(th.load(cfg.network_config['vgg_path']))
        self.encoder = get_vgg19(cfg.network_config['vgg_layer'], cfg.network_config['num_of_layer'], vgg.features)
        self.decoder = get_vgg19_decoder(cfg.network_config['decoder'], cfg.network_config['num_of_layer'])
        
        self.encoder_depth = len(self.encoder)
        self.decoder_depth = len(self.decoder)
        
        if init:
            for m in self.children():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode= 'fan_out', nonlinearity= 'relu')

        self.Normalized = Normalized
        if self.Normalized: 
            self.bn_list = nn.ModuleList([nn.BatchNorm2d(args[0]) for args in cfg.network_config['decoder']])
        
        self.get_layer_output = get_layer_output
        if self.get_layer_output:                      
            self.elist = []
            self.dlist = []    
            
    def forward(self, x): 
        if self.get_layer_output:
            self.elist.clear()
            self.dlist.clear()

        for i in range(self.encoder_depth):
            if self.get_layer_output:
                self.elist.append(x)
            x = self.encoder[i](x)

        for i in range(self.decoder_depth):
            if self.Normalized:
                x = self.bn_list[i](x)
            x = self.decoder[i](x)
            if self.get_layer_output:
                self.dlist.append(x)

        if self.get_layer_output:        
            self.dlist.reverse()
        return x
    
    def get_encoder_output(self):
        if self.get_layer_output:
            return self.elist
        return None
    
    def get_decoder_output(self):
        if self.get_layer_output:
            return self.dlist
        return None
    
    def save_decoder(self, save_path: str):
        th.save(self.decoder.state_dict(), save_path)

class Decoder_Train_Net(nn.Module):
    def __init__(self) -> None:
        super(Decoder_Train_Net, self).__init__()
        vgg = model.vgg19()
        vgg.load_state_dict(th.load(cfg.network_config['vgg_path']))
        self.encoder = get_vgg19(cfg.network_config['vgg_layer'], cfg.network_config['num_of_layer'], vgg.features)
        self.decoder = get_vgg19_decoder(cfg.network_config['decoder'], cfg.network_config['num_of_layer'])
        self.norm = get_InstanceNorm(cfg.network_config['decoder'], cfg.network_config['num_of_layer'])
        
        self.encoder_depth = len(self.encoder)
        self.decoder_depth = len(self.decoder)
        self.norm_depth = len(self.norm)

    def forward(self, x):
        pass
class VGG_Encoder(nn.Module): 
    def __init__(self):
        super(VGG_Encoder, self).__init__()
        vgg = model.vgg19()
        vgg.load_state_dict(th.load(cfg.network_config['vgg_path']))
        self.encoder = get_vgg19(cfg.network_config['vgg_layer'], cfg.network_config['num_of_layer'], vgg.features)

    def forward(self, x): 
        elist = list()
        for i in range(self.encoder.__len__()):
            x = self.encoder[i](x)
            elist.append(x)
        elist.reverse()
        return x, elist

class Style_Decoder(nn.Module):
    def __init__(self, decoder_path: str, styleTransCFG: list): 
        super(Style_Decoder, self).__init__()
        self.decoder = get_vgg19_decoder(cfg.network_config['decoder'], cfg.network_config['num_of_layer'])
        self.decoder.load_state_dict(th.load(decoder_path))
        self.decoder.requires_grad_(False)

        self.styleList = nn.ModuleList([net.get_style_trans(*block) for block in styleTransCFG])
        assert self.decoder.__len__() == self.styleList.__len__()

    def forward(self, x, eclist: list, eslist: list):
        assert len(eclist) == self.decoder.__len__()
        dlist = list()
        for i in range(self.decoder.__len__()):
            x = self.styleList[i](eclist[i], eslist[i], x)
            x = self.decoder[i](x)
            dlist.append(x)
        return x, dlist

class Style_Trans_Train(nn.Module):
    def __init__(self): 
        super(Style_Trans_Train, self).__init__()       
        self.encoder = VGG_Encoder()
        self.decoder = Style_Decoder(cfg.network_config['decoder_path'], cfg.style_config)
        
        self.eclist = None
        self.eslist = None
        self.dlist  = None
    
    def forward(self, c, s): 
        x, self.eclist = self.encoder(c)
        _, self.eslist = self.encoder(s)
        x, self.dlist  = self.decoder(x, self.eclist, self.eslist)
        return x

# ------------- MobileNetv2 -----------------
# refrence from https://gitee.com/xuyangyan/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py

def get_ConvNormActivation(in_channals, out_channals, kernel_size= 3, 
                        stride= 1, padding= 1, groups= 1, bias= True):
    return nn.Sequential(nn.Conv2d(in_channals, out_channals, kernel_size, stride, padding, 
                                   groups= groups, bias= bias),
                         nn.BatchNorm2d(out_channals),
                         nn.ReLU6(inplace= True))

def get_TransposedConvNormActivation(in_channals, out_channals, kernel_size= 3, 
                                stride= 1, padding= 1, groups= 1, bias= True):
    assert stride in (1, 2)
    output_padding = 1 if stride == 2 else 0
    return nn.Sequential(
                         nn.ConvTranspose2d(in_channals, out_channals, kernel_size, stride, 
                                            padding= padding, output_padding= output_padding,
                                            groups= groups, bias= bias),
                         nn.BatchNorm2d(out_channals),
                         nn.ReLU6(inplace= True))
     
def get_CNA_Block(in_channals, out_channals, kernel_size= 3, 
                  stride= 1, padding= 1, groups= 1, bias= True, Transposed= False):
    if Transposed:
        return get_TransposedConvNormActivation(in_channals, out_channals, kernel_size, 
                                                stride, padding, groups, bias)
    return get_ConvNormActivation(in_channals, out_channals, kernel_size, 
                                  stride, padding, groups, bias)

class InverteResidual(nn.Module):
    def __init__(self, in_channals, out_channals, stride, expand_ratio, Transposed= False):
        super(InverteResidual, self).__init__()
        assert stride in (1, 2)
        
        # 计算隐藏维度时舍入整数
        hidden_dim = round((out_channals if Transposed and expand_ratio != 1 else in_channals) * expand_ratio)
        # 是否残差连接
        self.identity = stride == 1 and in_channals == out_channals

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                *get_CNA_Block(hidden_dim, hidden_dim, 3, stride, 1,
                               groups= hidden_dim, bias= False, Transposed= Transposed),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channals, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channals),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                *get_CNA_Block(in_channals, hidden_dim, 1, 1, 0,
                               bias= False, Transposed= Transposed),
                # dw
                *get_CNA_Block(hidden_dim, hidden_dim, 3, stride, 1,
                               groups= hidden_dim, bias= False, Transposed= Transposed),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channals, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channals),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2_Encoder(nn.Module):
    def __init__(self, preTrained= False):
        super(MobileNetV2_Encoder, self).__init__()
        
        layers = [get_CNA_Block(3, 32, kernel_size= 3, stride= 2, bias= False)]
        block = InverteResidual
        in_channals = 32
        for t, c, n, s in cfg.mobv2_encoder_cfg:
            out_channals = c
            for i in range(n):
                layers.append(block(in_channals, out_channals, s if i == 0 else 1, t))
                in_channals = out_channals
        self.features = nn.Sequential(*layers)
        
        if preTrained:
            self.load_state_dict(th.load(cfg.mobv2_pretrained_model_path), strict= False)     
        self.features = nn.ModuleList([*self.features])
        self.ex_layer = cfg.mobv2_encoder_extract_layer
        
    def forward(self, c, s):
        en_s_tensors = []
        for i, layer in enumerate(self.features):
            c = layer(c)
            s = layer(s)
            if i in self.ex_layer:
                en_s_tensors.append(s)
        # en_s_tensors.append(s)
        en_s_tensors.reverse()
        return c, en_s_tensors
        
class MobileNetV2_Decoder(nn.Module):
    def __init__(self):
        super(MobileNetV2_Decoder, self).__init__()
        
        layers = [get_CNA_Block(32, 3, kernel_size= 3, stride= 2, bias= False, Transposed= True)]
        block = InverteResidual    
        out_channals = 32   
        for t, c, n, s in cfg.mobv2_encoder_cfg:
            in_channals = c
            for i in range(n):
                layers.append(block(in_channals, out_channals, s if i == 0 else 1, t, Transposed= True))
                out_channals = in_channals
        layers.reverse()
        # for l in layers:
        #     print(l)
        self.features = nn.ModuleList(layers)
        
        self.style_layers = nn.ModuleList([styleblock.get_style_block(block_name) for block_name in cfg.mobv2_style_block_cfg])
        self.in_layer = cfg.mobv2_decoder_insert_layer
        
    def forward(self, c, en_s_tensors):
        style_layer = iter(self.style_layers)
        style_tensors = iter(en_s_tensors)
        for i, layer in enumerate(self.features):
            if i in self.in_layer:
                # sl = next(style_layer)
                # st = next(style_tensors)
                # print(c.size(), st.size())
                # c = sl(c, st)
                c = next(style_layer)(c, next(style_tensors))
            # print(c.size())
            c = layer(c) 
        return c
    
class EstNet(nn.Module):
    def __init__(self, load_preTrained_model: Optional[str] = None):
        super(EstNet, self).__init__()
        
        if load_preTrained_model:
            self.Encoder = MobileNetV2_Encoder()
            self.Decoder = MobileNetV2_Decoder()
            self.load_state_dict(th.load(load_preTrained_model))
        else:
            self.Encoder = MobileNetV2_Encoder(preTrained= True)
            self.Decoder = MobileNetV2_Decoder()
        
    def forward(self, c, s):
        c, en_s_tensors = self.Encoder(c, s)
        x = self.Decoder(c, en_s_tensors)
        return x
        

if __name__ == '__main__':
    # print(Decoder_Train_Net())
    # mobv2 = model.mobilenet_v2(pretrained= True)
    # mobv2 = MobileNetV2_Encoder()
    # mobv2.load_state_dict(th.load(r'BoneNetwork_models\mobilenetv2-c5e733a8.pth'), strict= False)
    # print(mobv2)
    
    # test = th.rand((2, 3, 224, 224))
    
    # test = get_ConvNormActivation(3, 3, kernel_size= 1, stride= 1, padding= 0, groups= 3)(test)
    # print(test.size())
    # test = get_TransposedConvNormActivation(3, 3, kernel_size= 1, padding= 0, stride= 1, groups= 3)(test)
    # print(test.size())
    
    # test = InverteResidual(3, 16, 1, 6)(test)
    # print(test.size())
    # test = InverteResidual(16, 3, 1, 6, True)(test)
    # print(test.size())
    
    c = th.rand((2, 3, 224, 224))
    s = th.rand((2, 3, 224, 224))
    
    x = EstNet()(c, s)
    print(x.size())
    
    print(x)