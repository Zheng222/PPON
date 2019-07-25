import argparse
import torch
import os
import skimage.io as sio
import numpy as np
import utils
import skimage.color as sc
from models import networks
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='PPON Test')
parser.add_argument('--test_hr_folder', type=str, default='Test_Datasets/Set5/',
                    help='the folder of the target images')
parser.add_argument('--test_lr_folder', type=str, default='Test_Datasets/Set5_LR/',
                    help='the folder of the input images')
parser.add_argument('--output_folder', type=str, default='result/Set5/')
parser.add_argument('--models', type=str, default='ckpt/PPON_G.pth',
                    help='models file to use')

parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscaling factor')
parser.add_argument('--only_y', action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
parser.add_argument('--isHR', action='store_true', default=True)
parser.add_argument("--which_model_G", type=str, default="ppon")
opt = parser.parse_args()

print(opt)
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please without --cuda")

filepath = opt.test_lr_folder
if filepath.split('/')[-2] == 'Set5_LR' or filepath.split('/')[-2] == 'Set14_LR':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_sr = np.zeros(len(filelist))
ssim_sr = np.zeros(len(filelist))

opt.is_train = False


model = networks.define_G(opt)
if isinstance(model, nn.DataParallel):
    model = model.module
model.load_state_dict(torch.load(opt.models), strict=True)
i = 0
for imname in filelist:
    if opt.isHR:
        im_gt = sio.imread(opt.test_hr_folder + imname.split('/')[-1])
        im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    im_l = sio.imread(imname)
    if len(im_l.shape) < 3:
        if opt.isHR:
            im_gt = im_gt[..., np.newaxis]
            im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)

    if im_l.shape[2] > 3:
        if opt.isHR:
            im_gt = im_gt[..., 0:3]
        im_l = im_l[..., 0:3]

    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if opt.cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    with torch.no_grad():
        out_c, out_s, out_p = model(im_input)
        out_c, out_s, out_p = out_c.cpu(), out_s.cpu(), out_p.cpu()

        out_img_c = out_c.detach().numpy().squeeze()
        out_img_c = utils.convert_shape(out_img_c)

        out_img_s = out_s.detach().numpy().squeeze()
        out_img_s = utils.convert_shape(out_img_s)

        out_img_p = out_p.detach().numpy().squeeze()
        out_img_p = utils.convert_shape(out_img_p)

    if opt.isHR:
        if opt.only_y is True:
            im_label = utils.quantize(sc.rgb2ycbcr(im_gt)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(out_img_c)[:, :, 0])
        else:
            im_label = im_gt
            im_pre = out_img_c

        psnr_sr[i] = utils.compute_psnr(utils.shave(im_label, opt.upscale_factor),
                                        utils.shave(im_pre, opt.upscale_factor))
        ssim_sr[i] = utils.compute_ssim(utils.shave(im_label, opt.upscale_factor),
                                        utils.shave(im_pre, opt.upscale_factor))
    i += 1

    output_c_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + '_c.png')
    output_s_folder = os.path.join(opt.output_folder,
                                   imname.split('/')[-1].split('.')[0] + '_s.png')
    output_p_folder = os.path.join(opt.output_folder,
                                   imname.split('/')[-1].split('.')[0] + '_p.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    sio.imsave(output_c_folder, out_img_c)
    sio.imsave(output_s_folder, out_img_s)
    sio.imsave(output_p_folder, out_img_p)
    print('===> Saved {}-th image'.format(i))

print('Mean PSNR for SR: {}'.format(np.mean(psnr_sr)))
print('Mean SSIM for SR: {}'.format(np.mean(ssim_sr)))
