network_config = {
    'vgg_layer': (0, 2, 7, 12, 21, 30),
    'decoder': (
        (512, 512, 3, True), # relu5_1 
        (512, 256, 3, True), # relu4_1
        (256, 128, 1, True), # relu3_1
        (128,  64, 1, True), # relu2_1
        ( 64,   3, 0, False) # relu1_1
    ), 
    'num_of_layer': 5, 
    'vgg_path': r"./pretrained_models\vgg19-dcbb9e9d.pth", 
    'decoder_path': r"./pretrained_models\decoder1.4.pth"
}

style_config = [
    ['adain', None], 
    ['adain', None],
    ['adain', None],
    ['adain', None],
    ['adain', None],
]

# train_config = {
#     'lr': 0.001,
#     'opt': 'Adam',
#     'batch_size': 1, 
#     'samples': 400, 
#     'epochs': 100
# }

mobv2_pretrained_model_path = r'BoneNetwork_models\mobilenetv2-c5e733a8.pth'

mobv2_encoder_cfg = (
    # t, c, n, s
    (1,  16, 1, 1),
    (6,  24, 2, 2),
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    # (6, 320, 1, 1),
)

mobv2_encoder_extract_layer = (4, 7, 11, 17)

mobv2_style_block_cfg = (
    'adain',
    'adain', 
    'adain', 
    'adain', 
    'adain',  
)