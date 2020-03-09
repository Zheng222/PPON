import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import models.networks as networks
from .base_model import BaseModel
from models.loss import GANLoss, MultiscaleL1Loss
import pytorch_msssim

# args.is_train | pixel_weight | pixel_criterion | feature_weight | feature_criterion | gan_type | gan_weight
# lr_G | lr_D | lr_scheme | lr_steps | lr_gamma | ssim_weight | pretrained_model_G | pretrained_model_D

class PPONModel(BaseModel):
    def __init__(self, args):
        super(PPONModel, self).__init__(args)

        # define networks and load pre-trained models
        self.netG = networks.define_G(args).cuda()
        if self.is_train:
            if args.which_model == 'perceptual':
                self.netD = networks.define_D().cuda()
                self.netD.train()
            self.netG.train()

        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if args.pixel_weight > 0:
                l_pix_type = args.pixel_criterion
                if l_pix_type == 'l1':  # loss pixel type
                    self.cri_pix = nn.L1Loss().cuda()
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().cuda()
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = args.pixel_weight
            else:
                print('Remove pixel loss.')
                self.cri_pix = None  # critic pixel

            # G structure loss
            if args.structure_weight > 0:
                self.cri_msssim = pytorch_msssim.MS_SSIM(data_range=args.rgb_range).cuda()
                self.cri_ml1 = MultiscaleL1Loss().cuda()
            else:
                print('Remove structure loss.')
                self.cri_msssim = None
                self.cri_ml1 = None

            # G feature loss
            if args.feature_weight > 0:
                l_fea_type = args.feature_criterion
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().cuda()
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().cuda()
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = args.feature_weight
            else:
                print('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.vgg = networks.define_F().cuda()

            if args.gan_weight > 0:
                # gan loss
                self.cri_gan = GANLoss(args.gan_type, 1.0, 0.0).cuda()
                self.l_gan_w = args.gan_weight
            else:
                self.cri_gan = None

            # optimizers
            # G
            if args.which_model == 'structure':
                for param in self.netG.CFEM.parameters():
                    param.requires_grad = False
                for param in self.netG.CRM.parameters():
                    param.requires_grad = False

            if args.which_model == 'perceptual':
                for param in self.netG.CFEM.parameters():
                    param.requires_grad = False
                for param in self.netG.CRM.parameters():
                    param.requires_grad = False
                for param in self.netG.SFEM.parameters():
                    param.requires_grad = False
                for param in self.netG.SRM.parameters():
                    param.requires_grad = False
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('Warning: params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=args.lr_G)
            self.optimizers.append(self.optimizer_G)

            # D
            if args.which_model == 'perceptual':
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr_D)
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if args.lr_scheme == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer,
                                                                    args.lr_steps, args.lr_gamma))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        print('------------- Model initialized -------------')
        self.print_network()
        print('---------------------------------------------')

    def feed_data(self, data, need_HR=True):  # data = [LR, HR]
        self.var_L = data[0].cuda()

        if need_HR:  # train or val
            self.var_H = data[1].detach().cuda()

    def optimize_parameters(self):
        # G
        if self.args.which_model == 'perceptual':
            for p in self.netD.parameters():
                p.requires_grad = False
        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)

        l_g_total = 0

        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix

        if self.cri_msssim:
            l_g_mssim = 1.0 - self.cri_msssim(self.fake_H, self.var_H)
            l_g_total += l_g_mssim

        if self.cri_ml1:
            l_g_ml1 = self.cri_ml1(self.fake_H, self.var_H)
            l_g_total += l_g_ml1

        if self.cri_fea:  # vgg feature matching loss
            real_fea = self.vgg(self.var_H).detach()
            fake_fea = self.vgg(self.fake_H)


            vgg_loss = self.cri_fea(fake_fea, real_fea)
            l_g_fea = self.l_fea_w * vgg_loss
            l_g_total += l_g_fea

        # RelativisticGAN
        if self.cri_gan:
            pred_g_fake = self.netD(self.fake_H, self.fake_H)
            pred_g_real = self.netD(self.var_H, self.var_H)
            pred_g_real.detach_()
            l_g_gan = self.l_gan_w * (self.cri_gan(pred_g_real - torch.mean(pred_g_fake), False) +
                                      self.cri_gan(pred_g_fake - torch.mean(pred_g_real), True)) / 2
            l_g_total += l_g_gan


        l_g_total.backward()
        self.optimizer_G.step()

        if self.args.which_model == 'perceptual':
            # D
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            pred_d_real = self.netD(self.var_H, self.var_H)
            pred_d_fake = self.netD(self.fake_H.detach(), self.fake_H.detach())  # detach to avoid BP to G
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

            l_d_total = (l_d_real + l_d_fake) / 2
            l_d_total.backward()
            self.optimizer_D.step()

        # set log
        # G
        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()

        if self.args.structure_weight:
            self.log_dict['l_g_ml1'] = l_g_ml1.item()
            self.log_dict['l_g_msssim'] = l_g_mssim.item()

        if self.args.which_model == 'perceptual':
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_gan:
                self.log_dict['l_g_gan'] = l_g_gan.item()

            # D
            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()

            # D outputoptimize_parametsers
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False
        self.fake_H = self.netG(self.var_L)
        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)  # s--> the str version of network, n--> parameters
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '------------------- Generator --------------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

            # Discriminator
            if self.args.which_model == 'perceptual':
                s, n = self.get_network_description(self.netD)
                print('Number of parameters in D: {:,d}'.format(n))
                message = '\n\n\n-------------- Discriminator -------------\n' + s + '\n'
                with open(network_path, 'a') as f:
                    f.write(message)

            if self.cri_fea:  # F, Preceptual Network
                s, n = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n---------------- Perceptual Network ---------------\n' + s + '\n'
                with open(network_path, 'a') as f:
                    f.write(message)

    def load(self):
        load_path_G = self.args.pretrained_model_G
        if load_path_G is not None:
            print('loading models for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=False)

        load_path_D = self.args.pretrained_model_D
        if self.args.is_train and load_path_D is not None:
            print('loading models for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.cri_gan:
            self.save_network(self.save_dir, self.netD, 'D', iter_label)
