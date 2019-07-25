import math
import torch
import torch.nn as nn
from . import block as B
import torchvision

#######################
# Generator
#######################

class PPON(nn.Module):
    def __init__(self, in_nc, nf, nb, out_nc, upscale=4, act_type='lrelu'):
        super(PPON, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)  # common
        rb_blocks = [B.RRBlock_32() for _ in range(nb)]  # L1
        LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        ssim_branch = [B.RRBlock_32() for _ in range(2)]  # SSIM
        gan_branch = [B.RRBlock_32() for _ in range(2)]  # Gan

        upsample_block = B.upconv_block

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_ssim = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_gan = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            upsampler_ssim = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            upsampler_gan = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        HR_conv0_S = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1_S = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        HR_conv0_P = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1_P = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.CFEM = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)))
        self.SFEM = B.sequential(*ssim_branch)
        self.PFEM = B.sequential(*gan_branch)

        self.CRM = B.sequential(*upsampler, HR_conv0, HR_conv1)  # recon l1
        self.SRM = B.sequential(*upsampler_ssim, HR_conv0_S, HR_conv1_S)  # recon ssim
        self.PRM = B.sequential(*upsampler_gan, HR_conv0_P, HR_conv1_P)  # recon gan

    def forward(self, x):
        out_CFEM = self.CFEM(x)
        out_c = self.CRM(out_CFEM)

        out_SFEM = self.SFEM(out_CFEM)
        out_s = self.SRM(out_SFEM) + out_c

        out_PFEM = self.PFEM(out_SFEM)
        out_p = self.PRM(out_PFEM) + out_s

        return out_c, out_s, out_p

#########################
# Discriminator
#########################

class Discriminator_192(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='lrelu'):
        super(Discriminator_192, self).__init__()

        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type)  # 3-->64
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type,  # 64-->64, 96*96
                             act_type=act_type)

        conv2 = B.conv_block(base_nf, base_nf * 2, kernel_size=3, stride=1, norm_type=norm_type,  # 64-->128
                             act_type=act_type)
        conv3 = B.conv_block(base_nf * 2, base_nf * 2, kernel_size=4, stride=2, norm_type=norm_type,  # 128-->128, 48*48
                             act_type=act_type)

        conv4 = B.conv_block(base_nf * 2, base_nf * 4, kernel_size=3, stride=1, norm_type=norm_type,  # 128-->256
                             act_type=act_type)
        conv5 = B.conv_block(base_nf * 4, base_nf * 4, kernel_size=4, stride=2, norm_type=norm_type,  # 256-->256, 24*24
                             act_type=act_type)

        conv6 = B.conv_block(base_nf * 4, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type,  # 256-->512
                             act_type=act_type)
        conv7 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type,  # 512-->512 12*12
                             act_type=act_type)

        conv8 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type,  # 512-->512
                             act_type=act_type)
        conv9 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type,  # 512-->512 6*6
                             act_type=act_type)
        conv10 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type,
                              act_type=act_type)
        conv11 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type,  # 3*3
                              act_type=act_type)

        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
                                     conv9, conv10, conv11)

        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier2(x)
        return x

#########################
# Perceptual Network
#########################

# data range [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output