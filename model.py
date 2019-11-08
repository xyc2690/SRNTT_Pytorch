import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from collections import OrderedDict
import arch_util as arch_util
import time

class VGGMOD(nn.Module):

    def __init__(self):
        super(VGGMOD, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x1_1 = self.relu1_1(x)
        x = self.conv1_2(x1_1)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x2_1 = self.conv2_1(x)
        x = self.relu2_1(x2_1)
        x = self.conv2_2(x)
        x = self.pool1(x)
        x = self.conv3_1(x)
        x3_1 = self.relu3_1(x)
        return x3_1, x2_1, x1_1


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.lrelu(x)
        return x


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.tanh = nn.Tanh()
        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)
        out = self.lrelu(self.conv_second(out))
        content_fea = fea + out
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(content_fea)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(content_fea)))

        # out = self.conv_last(self.lrelu(self.HRconv(out)))
        # out = self.conv_last(self.tanh(self.HRconv(out)))
        out = self.tanh(self.conv_last(self.lrelu(self.HRconv(out))))

        # base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        # out += base
        return content_fea, out


class TextureTransfer(nn.Module):
    def __init__(self):
        super(TextureTransfer, self).__init__()
        self.name = 'TextureTransfer'
        # map_in concat map_ref (small)
        self.conv1_1 = nn.Conv2d(256+64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=64)
        self.residual1 = arch_util.make_layer(basic_block, 16)

        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        # upscaling 2x
        # self.conv1_3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True)
        # self.sub_pixel_conv1 = UpsampleBLock(256, 64, 2)
        self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # map_in concat map_ref (medium)
        self.conv2_1 = nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.residual2 = arch_util.make_layer(basic_block, 8)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)



        # upscaling 2x
        # self.conv2_3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=True)
        # self.sub_pixel_conv2 = UpsampleBLock(256, 64, 2)
        self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)


        # map_in concat map_ref (large)
        self.conv3_1 = nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.residual3 = arch_util.make_layer(basic_block, 4)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # output
        self.conv3_3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_4 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def _make_layer(self, block, num_blocks):
        layer = [block(64) for _ in range(num_blocks)]
        return nn.Sequential(*layer)

    def forward(self, map_in, map_ref):
        # small
        x = torch.cat((map_in, map_ref[0]), 1)
        input1 = self.lrelu(self.conv1_1(x))
        x = self.residual1(input1)
        x = self.conv1_2(x)
        x = self.bn1(x)
        feature_small = torch.add(x, map_in)

        # upscaling (2x)
        # x = self.conv1_3(feature_small)
        # small = self.sub_pixel_conv1(x)
        small = self.lrelu(self.pixel_shuffle(self.upconv1(feature_small)))

        # medium
        x = torch.cat((small, map_ref[1]), 1)
        input2 = self.lrelu(self.conv2_1(x))
        x = self.residual2(input2)
        x = self.conv2_2(x)
        x = self.bn2(x)
        feature_medium = torch.add(x, small)

        # upscaling (2x)
        # x = self.conv2_3(feature_medium)
        # medium = self.sub_pixel_conv2(x)
        medium = self.lrelu(self.pixel_shuffle(self.upconv2(feature_medium)))

        # large
        x = torch.cat((medium, map_ref[2]), 1)
        input3 = self.lrelu(self.conv3_1(x))
        x = self.residual3(input3)
        x = self.conv3_2(x)
        x = self.bn3(x)
        feature_large = torch.add(x, medium)

        x = self.conv3_3(feature_large)
        output = torch.tanh(self.conv3_4(x))

        return output


class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()
        self.name = 'SR'
        # self.content = Content()
        self.content = MSRResNet()
        self.TextureTransfer = TextureTransfer()
        self.load_path = 'SRGAN_100000.pth'


        self.load_net(self.load_path, self.content)
        for param in self.content.parameters():
            param.requires_grad = False

    def load_net(self, load_path, network):
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=True)

    def forward(self, batch_input, batch_maps):
        s = time.time()
        content_feature, upscale = self.content(batch_input)
        # t1 = time.time()
        output = self.TextureTransfer(content_feature, batch_maps)
        # t2 = time.time()
        # print('t1: ', t1-s, 't2: ', t2-t1)

        return upscale, output


class Discriminator(nn.Module):
    def __init__(self, l=0.2):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(l),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(l),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(l),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(l),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(l),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(l),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(l),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(l),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(l),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        # print ('D input size :' +  str(x.size()))
        y = self.net(x)
        # print ('D output size :' +  str(y.size()))
        return y.view(y.size()[0])


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

