import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import os
import time
from glob import glob
from collections import OrderedDict
from os import makedirs, environ
from os.path import join, exists, split, isfile
from scipy.misc import imread, imresize, imsave, imrotate

from loss import GANLoss, gram_matrix
cri_gan = GANLoss('gan', 1.0, 0.0)
from model import VGGMOD, SR, Discriminator, compute_gradient_penalty

torch.set_default_dtype(torch.float32)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# some global variables
MODEL_FOLDER = 'model'
SAMPLE_FOLDER = 'sample'

input_dir = 'sr_data/CUFED_128/input'  # original images
ref_dir = 'sr_data/CUFED_128/ref'  # reference images
map_dir = 'sr_data/CUFED_128/map_321' # texture maps after texture swapping

use_gpu = True
use_train_ref = True
pre_load_img = True

use_advloss = False
use_textloss = False
use_perceptual_loss = False

num_subset = 128
input_size = 40
num_init_epochs = 100
batch_size = 16

model = SR()
vggmodel = VGGMOD()
vggmodel.load_state_dict(torch.load('VGGMOD.pth'))
model = DataParallel(model)
mse_loss = nn.MSELoss()

netD = Discriminator()
netD = DataParallel(netD)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4)

if use_gpu:
    model.cuda()
    vggmodel.cuda()
    netD.cuda()

vggmodel.eval()

# check input dir
files_input = sorted(glob(join(input_dir, '*.png')))
files_map = sorted(glob(join(map_dir, '*.npz')))
files_ref = sorted(glob(join(ref_dir, '*.png')))
num_files = len(files_input)

if num_subset is not None:
    files_input = files_input[:num_subset]
    files_map = files_map[:num_subset]
    files_ref = files_ref[:num_subset]
    num_files = len(files_input)

assert num_files == len(files_ref) == len(files_map)

idx = np.arange(num_files)
num_batches = int(num_files / batch_size)

if pre_load_img:
    print('loading maps...')
    t0 = time.time()
    batch_maps_all = []
    for i in idx:
        if i % 10 == 0:
            print('[%04d/%04d]' % (i, num_files))
        batch_maps_all.append(np.load(files_map[i], allow_pickle=True)['target_map'])
    t1 = time.time()
    print('using: ', t1-t0, ' / ', (t1-t0)/num_files)
    print('loading maps fnished...')


# Save init model
save_path = 'Train/train_CUFED_128'

if not exists(save_path):
    makedirs(save_path)


#######
#       Reg, Gan, Text, Perceptual
# weight = [1, 1e-3, 5e-5, 1e-4]
weight = [1, 0, 0, 0]

torch.save(model.state_dict(), save_path + '/sr_model_' + str(0) + '.pth')
print('Start Training:')
for epoch in range(num_init_epochs):
    np.random.shuffle(idx)
    for n_batch in range(num_batches):
        step_time = time.time()
        sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
        batch_imgs = [imread(files_input[i], mode='RGB') for i in sub_idx]
        batch_truth = [img.astype(np.float32)/127.5-1 for img in batch_imgs]
        batch_input = [imresize(img, .25, interp='bicubic').astype(np.float32)/127.5-1 for img in batch_imgs]

        # t1 = time.time()
        # print('1_time: ', t1-step_time)
        if not pre_load_img:
            batch_maps_tmp = [np.load(files_map[i], allow_pickle=True)['target_map'] for i in sub_idx]
            t_load = time.time()
            print('Load time: ', t_load-step_time)
        else:
            batch_maps_tmp = [batch_maps_all[i] for i in sub_idx]

        batch_maps = [[] for _ in range(len(batch_maps_tmp[0]))]
        for s in batch_maps_tmp:
            for i, item in enumerate(batch_maps):
                item.append(s[i])

        if use_gpu:
            batch_maps = [torch.Tensor(np.array(b).astype(np.float32)).permute(0, 3, 1, 2).cuda() for b in batch_maps]
            batch_input = torch.tensor(np.array(batch_input)).permute(0, 3, 1, 2).cuda()
            batch_truth = torch.tensor(np.array(batch_truth)).permute(0, 3, 1, 2).cuda()
        else:
            batch_maps = [torch.tensor(b).float().permute(0, 3, 1, 2) for b in batch_maps]
            batch_input = torch.tensor(batch_input).permute(0, 3, 1, 2)
            batch_truth = torch.tensor(batch_truth).permute(0, 3, 1, 2)
            
        ##########################################################################################
        # # Train with ref #######################################################################
        ##########################################################################################
        if use_train_ref:
            l_total = 0
            l_d_real = 0
            l_d_fake = 0

            if use_advloss:
                # Train D twice
                for p in netD.parameters():
                    p.requires_grad = True
                for x in range(2):
                    upscale, output = model(batch_input, batch_maps)
                    optimizerD.zero_grad()
                    pred_d_fake = netD(output.detach()).detach()
                    pred_d_real = netD(batch_truth)
                    l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
                    l_d_real.backward(retain_graph=True)
                    pred_d_fake = netD(output.detach())
                    l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
                    l_d_fake.backward()
                    optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False

            # Loss Rec
            optimizer.zero_grad()
            upscale, output = model(batch_input, batch_maps)
            l_rec = mse_loss(output, batch_truth)
            l_total += l_rec * weight[0]

            # Train G
            l_g = 0
            if use_advloss:
                gan_type = 'ragan'
                if gan_type == 'gan':
                    pred_g_fake = netD(output)
                    l_g = cri_gan(pred_g_fake, True)

                elif gan_type == 'ragan':
                    pred_d_real = netD(batch_truth).detach()
                    pred_g_fake = netD(output)
                    l_g = (cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                           cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_total += l_g * weight[1]

            # Loss Texture
            l_tex = 0
            if use_textloss:
                vgg_outputs = vggmodel((output+1)*127.5)
                loss1 = mse_loss(gram_matrix(batch_maps[2]), gram_matrix(vgg_outputs[2])) / 4. \
                                    / ((input_size * input_size * 256) ** 2)
                loss2 = mse_loss(gram_matrix(batch_maps[1]), gram_matrix(vgg_outputs[1])) / 4.\
                                    / ((input_size * input_size * 512) ** 2)
                loss3 = mse_loss(gram_matrix(batch_maps[0]), gram_matrix(vgg_outputs[0])) / 4.\
                                    / ((input_size * input_size * 1024) ** 2)
                l_tex = (loss1 + loss2 + loss3) / 3
            l_total += l_tex * weight[2]

            with torch.no_grad():
                vgg_outputs_hr = vggmodel((batch_truth+1)*127.5)

            # Loss Perceptual
            l_per = 0
            if use_perceptual_loss:
                l_per3 = mse_loss(vgg_outputs[0], vgg_outputs_hr[0].detach())
                l_per2 = mse_loss(vgg_outputs[1], vgg_outputs_hr[1].detach())
                l_per1 = mse_loss(vgg_outputs[2], vgg_outputs_hr[2].detach())
                l_per = (l_per3 + l_per2 + l_per1) / 3
            l_total += l_per * weight[3]

            l_total.backward()
            optimizer.step()
        # print('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\n loss: %.4f rec_loss: %4f Per_loss: %4f G_loss: %4f l_d_real: %4f l_d_fake: %4f' %
        #       (epoch + 1, num_init_epochs, n_batch+1, num_batches, l_total.item(), l_rec.item(), l_per.item(), l_g.item(), l_d_real.item(), l_d_fake.item()))
        print('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\n loss: %.4f rec_loss: %4f Per_loss: %4f G_loss: %4f l_d_real: %4f l_d_fake: %4f' %
              (epoch + 1, num_init_epochs, n_batch+1, num_batches, l_total.item(), l_rec.item(), l_per, l_g, l_d_real, l_d_fake))

        ##########################################################################################
        # # Train with Truth #####################################################################
        ##########################################################################################
        l_total = 0
        l_d_real = 0
        l_d_fake = 0

        if use_advloss:
            # Train D twice
            for p in netD.parameters():
                p.requires_grad = True
            for x in range(2):
                upscale, output = model(batch_input, vgg_outputs_hr)
                optimizerD.zero_grad()
                pred_d_fake = netD(output.detach()).detach()
                pred_d_real = netD(batch_truth)
                l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
                l_d_real.backward(retain_graph=True)
                pred_d_fake = netD(output.detach())
                l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
                l_d_fake.backward()
                optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False

        # Loss Rec
        optimizer.zero_grad()
        upscale, output = model(batch_input, vgg_outputs_hr)
        l_rec = mse_loss(output, batch_truth)
        l_total += l_rec * weight[0]

        l_g = 0
        # Train G
        if use_advloss:
            gan_type = 'ragan'
            if gan_type == 'gan':
                pred_g_fake = netD(output)
                l_g = cri_gan(pred_g_fake, True)

            elif gan_type == 'ragan':
                pred_d_real = netD(batch_truth).detach()
                pred_g_fake = netD(output)
                l_g = (cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_total += l_g * weight[1]

        # Loss Texture
        l_tex = 0
        if use_textloss:
            vgg_outputs = vggmodel((output + 1) * 127.5)
            loss1 = mse_loss(gram_matrix(vgg_outputs_hr[2]), gram_matrix(vgg_outputs[2])) / 4. \
                    / ((input_size * input_size * 256) ** 2)
            loss2 = mse_loss(gram_matrix(vgg_outputs_hr[1]), gram_matrix(vgg_outputs[1])) / 4. \
                    / ((input_size * input_size * 512) ** 2)
            loss3 = mse_loss(gram_matrix(vgg_outputs_hr[0]), gram_matrix(vgg_outputs[0])) / 4. \
                    / ((input_size * input_size * 1024) ** 2)
            l_tex = (loss1 + loss2 + loss3) / 3
        l_total += l_tex * weight[2]

        # Loss Perceptual
        l_per = 0
        if use_perceptual_loss:
            # vgg_outputs_hr = vggmodel((batch_truth + 1) * 127.5)
            l_per3 = mse_loss(vgg_outputs[0], vgg_outputs_hr[0].detach())
            l_per2 = mse_loss(vgg_outputs[1], vgg_outputs_hr[1].detach())
            l_per1 = mse_loss(vgg_outputs[2], vgg_outputs_hr[2].detach())
            l_per = (l_per3 + l_per2 + l_per1) / 3
        l_total += l_per * weight[3]

        l_total.backward()
        optimizer.step()

        # print('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\n loss: %.4f rec_loss: %4f Per_loss: %4f G_loss: %4f l_d_real: %4f l_d_fake: %4f' %
        #       (epoch + 1, num_init_epochs, n_batch+1, num_batches, l_total.item(), l_rec.item(), l_per.item(), l_g.item(), l_d_real.item(), l_d_fake.item()))
        print('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\n loss: %.4f rec_loss: %4f Per_loss: %4f G_loss: %4f l_d_real: %4f l_d_fake: %4f' %
              (epoch + 1, num_init_epochs, n_batch+1, num_batches, l_total.item(), l_rec.item(), l_per, l_g, l_d_real, l_d_fake))

    torch.save(model.state_dict(), save_path + '/sr_model_' + str(epoch+1) + '.pth')
    if use_advloss:
        torch.save(netD.state_dict(), save_path + '/d_model_' + str(epoch+1) + '.pth')




