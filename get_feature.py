import torch,os
import torch.nn as nn
import numpy as np
import copy
from scipy.misc import imread, imresize

from glob import glob
from os.path import exists, join, split, realpath, dirname
from os import makedirs

from swap import *
from model import VGGMOD
from torchvision import models
from torch.autograd import Variable




###############################################
# Create VGG19 conv1_1 <---> conv3-1 sub-model
###############################################
# vgg19 = models.vgg19(pretrained=True)
# vgg_mod = VGGMOD()
# vgg_dict = vgg19.state_dict()
# mod_dict = vgg_mod.state_dict()
# mod_dict['conv1_1.weight'] = vgg_dict['features.0.weight']
# mod_dict['conv1_1.bias'] = vgg_dict['features.0.bias']
# mod_dict['conv1_2.weight'] = vgg_dict['features.2.weight']
# mod_dict['conv1_2.bias'] = vgg_dict['features.2.bias']
# mod_dict['conv2_1.weight'] = vgg_dict['features.5.weight']
# mod_dict['conv2_1.bias'] = vgg_dict['features.5.bias']
# mod_dict['conv2_2.weight'] = vgg_dict['features.7.weight']
# mod_dict['conv2_2.bias'] = vgg_dict['features.7.bias']
# mod_dict['conv3_1.weight'] = vgg_dict['features.10.weight']
# mod_dict['conv3_1.bias'] = vgg_dict['features.10.bias']
# vgg_mod.load_state_dict(mod_dict)
# torch.save(vgg_mod.state_dict(), 'VGGMOD.pth')

model = VGGMOD()
model.load_state_dict(torch.load('VGGMOD.pth'))
model.cuda()
model.eval()
swaper = Swap()

data_folder = 'small'
data_folder = 'test'
data_folder = 'data/CUFED_64'
data_folder = '/hdd/sr_data/CUFED'
data_folder = '/hdd/sr_data/Flickr'



input_path = join(data_folder, 'input')
ref_path = join(data_folder, 'ref')
matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
save_path = join(data_folder, 'map_321')
if not exists(save_path):
    makedirs(save_path)

input_files = sorted(glob(join(input_path, '*.png')))
ref_files = sorted(glob(join(ref_path, '*.png')))
n_files = len(input_files)

input_size = 40


print_format = '%%0%dd/%%0%dd' % (len(str(n_files)), len(str(n_files)))
for i in range(n_files):
    file_name = join(save_path, split(input_files[i])[-1].replace('.png', '.npz'))
    if exists(file_name):
        continue
    print(print_format % (i + 1, n_files))
    img_in_lr = imresize(imread(input_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
    img_ref = imresize(imread(ref_files[i], mode='RGB'), (input_size * 4, input_size * 4), interp='bicubic')
    img_ref_lr = imresize(img_ref, (input_size, input_size), interp='bicubic')
    img_in_sr = imresize(img_in_lr, (input_size * 4, input_size * 4), interp='bicubic')
    img_ref_sr = imresize(img_ref_lr, (input_size * 4, input_size * 4), interp='bicubic')

    with torch.no_grad():
        map_in_sr, _, _ = model(torch.Tensor(img_in_sr).permute(2,0,1).unsqueeze(0).cuda())
        map_ref = model(torch.Tensor(img_ref).permute(2,0,1).unsqueeze(0).cuda())
        map_ref_sr, _, _ = model(torch.Tensor(img_ref_sr).permute(2,0,1).unsqueeze(0).cuda())

    # patch matching and swapping
    other_style = []
    for m in map_ref[1:]:
        other_style.append([m.cpu().numpy().squeeze().transpose(1, 2, 0)])
    map_in_sr = map_in_sr.cpu().numpy().squeeze().transpose(1, 2, 0)
    map_ref_sr = map_ref_sr.cpu().numpy().squeeze().transpose(1, 2, 0)

    map_ref = [k.cpu().numpy().squeeze().transpose(1, 2, 0) for k in map_ref]
    maps, weights, correspondence = swaper.conditional_swap_multi_layer(
        content=map_in_sr,
        style=[map_ref[0]],
        condition=[map_ref_sr],
        other_styles=other_style
    )
    # save maps
    np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)