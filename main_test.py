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

from loss import GANLoss
''
cri_gan = GANLoss('gan', 1.0, 0.0)

from model import VGGMOD, SR, Discriminator, compute_gradient_penalty


def gram_matrix(features):
    """
    Compute the Gram matrix from feature maps.

    Input: PyTorch Tensor of shape (N, C, H, W) representing feature maps for
           a batch of N images.
    Output: PyTorch Tensor of shape (N, C, C) representing the
            Gram matrices for the N images.
    """
    N = features.shape[0]
    C = features.shape[1]
    H = features.shape[2]
    W = features.shape[3]

    features = features.reshape(N, C, H * W)
    gram_matrix = torch.matmul(features, features.transpose(1, 2))

    # # Normalize
    # gram_matrix = gram_matrix / (2 * C * H * W)

    return gram_matrix


torch.set_default_dtype(torch.float32)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# some global variables
MODEL_FOLDER = 'model'
SAMPLE_FOLDER = 'sample'

input_dir = 'sr_data/CUFED/input'  # original images
ref_dir = 'sr_data/CUFED/ref'  # reference images
map_dir = 'sr_data/CUFED/map_321' # texture maps after texture swapping

use_train_ref = True

pre_load_img = False
num_subset = 128

input_size = 40
num_init_epochs = 100
batch_size = 16
use_gpu = True
use_advloss = True
use_textloss = True
use_perceptual_loss = True


model = SR()
vggmodel = VGGMOD()
vggmodel.load_state_dict(torch.load('VGGMOD.pth'))
model = DataParallel(model)
mse_loss = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

netD = Discriminator()
netD = DataParallel(netD)
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4)

if use_gpu:
    model.cuda()
    vggmodel.cuda()
    netD.cuda()

vggmodel.eval()
# vggmodel = DataParallel(vggmodel)

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

# Load pretrained MSE model
load_net = torch.load('sr_model_init.pth')
load_net_clean = OrderedDict()  # remove unnecessary 'module.'
# for k, v in load_net.items():
#     if k.startswith('module.'):
#         load_net_clean[k[7:]] = v
#     else:
#         load_net_clean[k] = v
model.load_state_dict(load_net, strict=True)

# Save init model
save_path = 'train_CUFED_128'

if not exists(save_path):
    makedirs(save_path)

torch.save(model.state_dict(), save_path + '/sr_model_' + str(0) + '.pth')

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
        # t2 = time.time()
        # print('2_readfile_time: ', t2-t1)
        batch_maps = [[] for _ in range(len(batch_maps_tmp[0]))]
        for s in batch_maps_tmp:
            for i, item in enumerate(batch_maps):
                item.append(s[i])
        # t3 = time.time()
        # print('3_time: ', t3-t2)
        if use_gpu:
            batch_maps = [torch.Tensor(np.array(b).astype(np.float32)).permute(0, 3, 1, 2).cuda() for b in batch_maps]
            # t4 = time.time()
            # print('4_time: ', t4 - t3)
            batch_input = torch.tensor(np.array(batch_input)).permute(0, 3, 1, 2).cuda()
            batch_truth = torch.tensor(np.array(batch_truth)).permute(0, 3, 1, 2).cuda()
        else:
            batch_maps = [torch.tensor(b).float().permute(0, 3, 1, 2) for b in batch_maps]
            batch_input = torch.tensor(batch_input).permute(0, 3, 1, 2)
            batch_truth = torch.tensor(batch_truth).permute(0, 3, 1, 2)
        # end_time = time.time()-step_time
        # print('end_time: ',end_time)
        t_proc = time.time()
        # print('proc time: ', t_proc - t_load)
        if use_train_ref:
            # # Train with ref #######################################################################
            for p in netD.parameters():
                p.requires_grad = False
            optimizer.zero_grad()
            upscale, output = model(batch_input, batch_maps)
            l_tex = 0
            l_g_gan = 0
            l_total = 0
            l_rec = mse_loss(output, batch_truth)

            l_total += l_rec

            if use_advloss:
                gan_type = 'ragan'
                if gan_type == 'gan':
                    pred_g_fake = netD(output)
                    l_g_gan = 1e-6 * cri_gan(pred_g_fake, True)

                elif gan_type == 'ragan':
                    pred_d_real = netD(batch_truth).detach()
                    pred_g_fake = netD(output)
                    l_g_gan = 1e-6 * (
                            cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                            cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            else:
                l_g_gan = 0

            l_total += l_g_gan

            if use_textloss:
                vgg_outputs = vggmodel((output+1)*127.5)
                loss1 = mse_loss(gram_matrix(batch_maps[2]), gram_matrix(vgg_outputs[2])) / 4. \
                                    / ((input_size * input_size * 256) ** 2)
                loss2 = mse_loss(gram_matrix(batch_maps[1]), gram_matrix(vgg_outputs[1])) / 4.\
                                    / ((input_size * input_size * 512) ** 2)
                loss3 = mse_loss(gram_matrix(batch_maps[0]), gram_matrix(vgg_outputs[0])) / 4.\
                                    / ((input_size * input_size * 1024) ** 2)

                l_tex = (loss1 + loss2 + loss3) / 3
                l_total += l_tex * 5e-5

            if use_perceptual_loss:
                vgg_outputs_hr = vggmodel((batch_truth+1)*127.5)
                l_per = mse_loss(vgg_outputs[0], vgg_outputs_hr[0].detach())
                l_total += l_per * 1e-4

            l_total.backward()
            optimizer.step()

            if use_advloss:
                # D
                for p in netD.parameters():
                    p.requires_grad = True

                optimizerD.zero_grad()
                pred_d_fake = netD(output.detach()).detach()
                pred_d_real = netD(batch_truth)
                l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
                l_d_real.backward()
                pred_d_fake = netD(output.detach())
                l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
                l_d_fake.backward()
                optimizerD.step()
            t_ref = time.time()
            print('ref time: ', t_ref - t_proc)

        # # Train with Truth ##########################################################################
        for p in netD.parameters():
            p.requires_grad = False
        optimizer.zero_grad()
        if not use_train_ref:
            vgg_outputs_hr = vggmodel((batch_truth+1)*127.5)

        upscale, output = model(batch_input, vgg_outputs_hr)
        l_tex = 0
        l_g_gan = 0
        l_total = 0
        l_rec = mse_loss(output, batch_truth)
        l_total += l_rec

        if use_advloss:
            gan_type = 'ragan'
            if gan_type == 'gan':
                pred_g_fake = netD(output)
                l_g_gan = 1e-6 * cri_gan(pred_g_fake, True)

            elif gan_type == 'ragan':
                pred_d_real = netD(batch_truth).detach()
                pred_g_fake = netD(output)
                l_g_gan = 1e-6 * (
                        cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        else:
            l_g_gan = 0

        l_total += l_g_gan

        if use_textloss:
            vgg_outputs = vggmodel((output+1)*127.5)
            loss1 = mse_loss(gram_matrix(batch_maps[2]), gram_matrix(vgg_outputs[2])) / 4. \
                                / ((input_size * input_size * 256) ** 2)
            loss2 = mse_loss(gram_matrix(batch_maps[1]), gram_matrix(vgg_outputs[1])) / 4.\
                                / ((input_size * input_size * 512) ** 2)
            loss3 = mse_loss(gram_matrix(batch_maps[0]), gram_matrix(vgg_outputs[0])) / 4.\
                                / ((input_size * input_size * 1024) ** 2)

            l_tex = (loss1 + loss2 + loss3) / 3
            l_total += l_tex * 5e-5

        if use_perceptual_loss:
            # vgg_outputs_hr = vggmodel((batch_truth+1)*127.5)
            l_per = mse_loss(vgg_outputs[0], vgg_outputs_hr[0].detach())
            l_total += l_per * 1e-4

        l_total.backward()
        optimizer.step()
        # *********************************************************************

        if use_advloss:
            # D
            for p in netD.parameters():
                p.requires_grad = True

            optimizerD.zero_grad()
            pred_d_fake = netD(output.detach()).detach()
            pred_d_real = netD(batch_truth)
            l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
            l_d_real.backward()
            pred_d_fake = netD(output.detach())
            l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
            l_d_fake.backward()
            optimizerD.step()

            print('Pre-train: Epoch [%02d/%02d] Batch [%03d/%03d]\n loss: %.4f rec_loss: %4f G_loss: %4f l_d_real: %4f l_d_fake: %4f' %
                  (epoch + 1, num_init_epochs, n_batch+1, num_batches, l_total.item(), l_rec.item(),l_g_gan.item(), l_d_real.item(), l_d_fake.item()))
        t_tru = time.time()
        # print('tre time: ', t_tru - t_ref)



    # Vis
    # if epoch % 20 == 0:
    #     print('epoch save input/output')
    #     save_path = 'vis/' + str(epoch)
    #     if not exists(save_path):
    #         makedirs(save_path)
    #     num = output.shape[0]
    #     np_input = batch_input.data.cpu().numpy().transpose(0, 2, 3, 1)
    #     np_output = output.data.cpu().numpy().transpose(0, 2, 3, 1)
    #     for ii in range(1, num):
    #         imsave(save_path + '/output_' + str(ii) + '.png', ((np_output[ii] + 1) * 127.5).astype(np.uint8))

    torch.save(model.state_dict(), save_path + '/sr_model_' + str(epoch+1) + '.pth')
    if use_advloss:
        torch.save(netD.state_dict(), save_path + '/d_model_' + str(epoch+1) + '.pth')




