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
    'decoder_path': r"./pretrained_models\decoder0.2.pth"
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