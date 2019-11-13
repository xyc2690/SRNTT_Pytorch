import numpy as np
from glob import glob
from os.path import exists, join, split, realpath, dirname
from os import makedirs

from model import VGGMOD,SR
from swap import *

from scipy.misc import imread, imresize
import argparse

scale = 4

parser = argparse.ArgumentParser('preprocess_maps')
parser.add_argument('--data_folder', type=str, default='sr_data/CUFED', help='The dir of dataset: CUFED')
args = parser.parse_args()

data_folder = args.data_folder
if 'CUFED' in data_folder:
    input_size = 40
else:
    raise Exception('Unrecognized dataset!')

input_path = join(data_folder, 'input')
ref_path = join(data_folder, 'ref')
matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
save_path = join(data_folder, 'map_321')
if not exists(save_path):
    makedirs(save_path)

input_files = sorted(glob(join(input_path, '*.png')))
ref_files = sorted(glob(join(ref_path, '*.png')))
n_files = len(input_files)
assert n_files == len(ref_files)

use_gpu = True

swaper = Swap()
vggmodel = VGGMOD()
srmodel = SR()

if use_gpu:
    vggmodel.cuda()
    srmodel.cuda()

with torch.no_grad():
    print_format = '%%0%dd/%%0 %dd' % (len(str(n_files)), len(str(n_files)))
    for i in range(n_files):
        file_name = join(save_path, split(input_files[i])[-1].replace('.png', '.npz'))
        if exists(file_name):
            continue
        print(print_format % (i + 1, n_files))
        img_in_lr = imresize(imread(input_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
        img_ref = imresize(imread(ref_files[i], mode='RGB'), (input_size*scale, input_size*scale), interp='bicubic')
        img_ref_lr = imresize(imread(ref_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
        img_input_sr = imresize(img_in_lr, 4.0, interp='bicubic')
        img_ref_sr = imresize(img_ref_lr, 4.0, interp='bicubic')

        # get feature maps via VGG19
        if use_gpu:
            map_in_sr, _, _ = vggmodel(torch.Tensor(img_input_sr).unsqueeze(0).permute(0,3,1,2).cuda())
            map_ref = vggmodel(torch.Tensor(img_ref).unsqueeze(0).permute(0,3,1,2).cuda())
            map_ref_sr, _, _ = vggmodel(torch.Tensor(img_ref_sr).unsqueeze(0).permute(0,3,1,2).cuda())
        else:
            map_in_sr, _, _ = vggmodel(torch.Tensor(img_input_sr).unsqueeze(0).permute(0,3,1,2))
            map_ref = vggmodel(torch.Tensor(img_ref).unsqueeze(0).permute(0,3,1,2))
            map_ref_sr, _, _ = vggmodel(torch.Tensor(img_ref_sr).unsqueeze(0).permute(0,3,1,2))

        # patch matching and swapping
        other_style = []
        for m in map_ref[1:]:
            other_style.append([m.cpu().numpy().squeeze().transpose(1, 2, 0)])
        map_ref_tmp = []
        for m in map_ref:
            map_ref_tmp.append([m.cpu().numpy().squeeze().transpose(1, 2, 0)])
        map_ref = map_ref_tmp
        map_in_sr = map_in_sr.cpu().numpy().squeeze().transpose(1, 2, 0)
        map_ref_sr = map_ref_sr.cpu().numpy().squeeze().transpose(1, 2, 0)

        maps, weights, correspondence = swaper.conditional_swap_multi_layer(
            content=map_in_sr,
            style=[map_ref[0]],
            condition=[map_ref_sr],
            other_styles=other_style
        )
        # save maps
        np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)
