import torch as th
import torch.nn as nn 
import torchvision.models as model
import networks as net
import config as cfg

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
        super(Decoder_Train_Net, self).__ini
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

if __name__ == '__main__':
    print(Decoder_Train_Net())