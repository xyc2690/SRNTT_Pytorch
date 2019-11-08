import numpy as np
import torch
import torch.nn as nn
import os
import time
from glob import glob
from os import makedirs, environ
from os.path import join, exists, split, isfile
from scipy.misc import imread, imresize, imsave, imrotate
import logging
from collections import OrderedDict
import cv2

from model import VGGMOD,SR
from swap import *

torch.set_default_dtype(torch.float32)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_dir = 'im1.png'
ref_dir = 'im2.png'

map_path = 'map.npz'

use_gpu = True
is_original_image = True

swaper = Swap()
vggmodel = VGGMOD()
srmodel = SR()

if use_gpu:
    vggmodel.cuda()
    srmodel.cuda()

# check input_dir
img_input, img_hr = None, None
if isinstance(input_dir, np.ndarray):
    assert len(input_dir.shape) == 3
    img_input = np.copy(input_dir)
elif isfile(input_dir):
    img_input = imread(input_dir, mode='RGB')
else:
    logging.error('Unrecognized input_dir %s' % input_dir)
    exit(0)

h, w, _ = img_input.shape
if is_original_image:
    # ensure that the size of img_input can be divided by 4 with no remainder
    h = int(h // 4 * 4)
    w = int(w // 4 * 4)
    img_hr = img_input[0:h, 0:w, ::]
    img_input = imresize(img_hr, .25, interp='bicubic')
    h, w, _ = img_input.shape
img_input_copy = np.copy(img_input)

# ref img
is_ref = True
if ref_dir is None:
    is_ref = False
    ref_dir = input_dir

img_ref = []

if not isinstance(ref_dir, (list, tuple)):
    ref_dir = [ref_dir]

for ref in ref_dir:
    if isinstance(ref, np.ndarray):
        assert len(ref.shape) == 3
        img_ref.append(np.copy(ref))
    elif isfile(ref):
        img_ref.append(imread(ref, mode='RGB'))
    else:
        logging.info('Unrecognized ref_dir type!')
        exit(0)

for i in range(len(img_ref)):
    h2, w2, _ = img_ref[i].shape
    h2 = int(h2 // 4 * 4)
    w2 = int(w2 // 4 * 4)
    img_ref[i] = img_ref[i][0:h2, 0:w2, ::]
    if not is_ref and is_original_image:
        img_ref[i] = imresize(img_ref[i], .25, interp='bicubic')

img_input_sr = imresize(img_input, 4.0, interp='bicubic')
img_ref_sr = []
for i in img_ref:
    img_ref_downscale = imresize(i, .25, interp='bicubic')
    img_ref_sr.append(imresize(img_ref_downscale, 4.0, interp='bicubic'))


vggmodel.load_state_dict(torch.load('VGGMOD.pth'))

# srmodel.load_state_dict(torch.load('init20/sr_model_19.pth'))

with torch.no_grad():
    if not exists(map_path):

        if use_gpu:
            map_in_sr, _, map_in_sr_2 = vggmodel(torch.Tensor(img_input_sr).unsqueeze(0).permute(0,3,1,2).cuda())
            map_ref = vggmodel(torch.Tensor(img_ref[0]).unsqueeze(0).permute(0,3,1,2).cuda())
            map_ref_sr, _, map_ref_sr_2 = vggmodel(torch.Tensor(img_ref_sr[0]).unsqueeze(0).permute(0,3,1,2).cuda())
        else:
            map_in_sr, _, _ = vggmodel(torch.Tensor(img_input_sr).unsqueeze(0).permute(0,3,1,2))
            map_ref = vggmodel(torch.Tensor(img_ref[0]).unsqueeze(0).permute(0,3,1,2))
            map_ref_sr, _, _ = vggmodel(torch.Tensor(img_ref_sr[0]).unsqueeze(0).permute(0,3,1,2))

        imsave('samples_test/inp_hr.png', img_hr)
        imsave('samples_test/inp_lr.png', img_input)
        imsave('samples_test/inp_sr.png', img_input_sr)
        imsave('samples_test/ref_hr.png', img_ref[0])
        imsave('samples_test/ref_sr.png', img_ref_sr[0])


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

        np.savez(map_path, target_map=maps, weights=[], correspondence=[])
    else:
        print('existing maps. Loading....')
        maps = np.load(map_path, allow_pickle=True)['target_map']

    if use_gpu:
        input_lr = torch.Tensor((img_input /127.5)-1).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        input_maps = [torch.tensor(b).float().unsqueeze(0).permute(0, 3, 1, 2).cuda() for b in maps]
    else:
        input_lr = torch.Tensor((img_input /127.5)-1).unsqueeze(0).permute(0, 3, 1, 2)
        input_maps = [torch.tensor(b).float().unsqueeze(0).permute(0, 3, 1, 2) for b in maps]


    # test model

    load_net = torch.load('sr_model_final.pth')
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
	for k, v in load_net.items():
		if k.startswith('module.'):
			load_net_clean[k[7:]] = v
		else:
			load_net_clean[k] = v
	srmodel.load_state_dict(load_net_clean, strict=True)
	upscale, output = srmodel(input_lr, input_maps)
	im_out = output.cpu().numpy().squeeze().transpose(1,2,0)
	im_up = upscale.cpu().numpy().squeeze().transpose(1,2,0)

	save_path = 'sample_train_CUFED_128'

	if not exists(save_path):
		makedirs(save_path)
	imsave(save_path + '/output_'+str(i)+'.png', ((im_out+1)*127.5).astype(np.uint8))
	
    imsave(save_path + '/upscale'+'.png', ((im_up+1)*127.5).astype(np.uint8))
    imsave(save_path + '/HR.png', img_hr)
    imsave(save_path + '/bicubic.png', img_input_sr)


