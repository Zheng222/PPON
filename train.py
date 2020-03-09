import argparse, os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models import PPON_model
import utils
from data import srdata
import skimage.color as sc
from data import val_data

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Training settings
parser = argparse.ArgumentParser(description="PPON")
parser.add_argument("--batch_size", type=int, default=25,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=600,
                    help="number of epochs to train")
# parser.add_argument("--lr", type=float, default=2e-4,
#                     help="Learning Rate. Default=2e-4")
# parser.add_argument("--step_size", type=int, default=200,
#                     help="learning rate decay per N epochs")
# parser.add_argument("--gamma", type=int, default=0.5,
#                     help="learning rate decay factor for step decay")
# parser.add_argument("--resume", default="", type=str,
#                     help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--test_every", type=int, default=138)
parser.add_argument("--n_train", type=int, default=3450,
                    help="number of training set")
parser.add_argument("--threads", type=int, default=16,
                    help="number of threads for data loading")
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
# parser.add_argument("--pretrained", default="", type=str,
#                     help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--noise_level", type=list, default=['G', 10])  # gaussian noise (level 10)

parser.add_argument("--which_model", type=str, default="content")  # content | structure | perceptual
parser.add_argument("--pixel_weight", type=float, default=1)
parser.add_argument("--pixel_criterion", type=str, default='l1')

parser.add_argument("--structure_weight", type=float, default=0)


parser.add_argument("--feature_weight", type=float, default=0)
parser.add_argument("--feature_criterion", type=str, default='l1')  # l1
parser.add_argument("--gan_type", type=str, default='gan')  # gan
parser.add_argument("--gan_weight", type=float, default=0)

parser.add_argument("--lr_G", type=float, default=2e-4)
parser.add_argument("--lr_D", type=float, default=2e-4)
parser.add_argument("--lr_scheme", type=str, default='MultiStepLR')
parser.add_argument("--lr_steps", type=list, default=[200, 300, 400, 500])
parser.add_argument("--lr_gamma", type=float, default=0.5)
parser.add_argument("--pretrained_model_G", type=str, default='ckpt/PPON_G.pth')
parser.add_argument("--pretrained_model_D", type=str, default=None)
parser.add_argument("--save_path", type=str, default='ckpt_stage1_noise10')

args = parser.parse_args()

is_y = True
print(args)
print("Random Seed: ", args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

print("===> Loading datasets")
trainset = srdata.df2k_data(args)
testset = val_data.DatasetFromFolderVal("/mnt/hz/datasets/Set5/",
                                         "/mnt/hz/datasets/Set5_LRDN/x{}/".format(args.scale),
                                         args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, batch_size=args.testBatchSize, shuffle=False)

print("===> Building models")
args.is_train = True
model = PPON_model.PPONModel(args)
model.print_network()
model.load()

print("===> Setting Optimizer")


def train(epoch):
    print('epoch =', epoch, 'lr = ', model.get_current_learning_rate())
    for iteration, batch in enumerate(training_data_loader, 1):

        model.feed_data(batch)
        model.optimize_parameters()

        if iteration % 100 == 0:
            logs = model.get_current_log()
            if args.which_model == 'content':
                print("===> Epoch[{}]({}/{}): l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                 logs['l_g_pix']))
            elif args.which_model == 'structure':
                print("===> Epoch[{}]({}/{}): msssim: {:.5f}, msl1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                                   logs['l_g_msssim'], logs['l_g_msl1']))
            elif args.which_model == 'perceptual':
                print("===> Epoch[{}]({}/{}): vgg: {:.5f}, l_g_gan: {:.5f}, l_d_fake: {:.5f}, l_d_real: {:.5f}".format(
                    epoch, iteration, len(training_data_loader), logs['l_g_fea'], logs['l_g_gan'], logs['l_d_fake'], logs['l_d_real']
                ))


def test():
    avg_psnr = 0

    for batch in testing_data_loader:
        input, target = batch[0].detach(), batch[1].detach()
        model.feed_data([input], need_HR=False)
        model.test()
        pre = model.get_current_visuals(need_HR=False)
        sr_img = utils.tensor2np(pre['SR'].data)
        gt_img = utils.tensor2np(target.data[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if is_y is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)


    print("===> Valid. psnr: {:.4f}".format(avg_psnr / len(testing_data_loader)))


def save_checkpoint(epoch):
    model.save(epoch)
    print("===> {:d}-th checkpoint is saved".format(epoch))


print("===> Training")
for epoch in range(args.start_epoch, args.nEpochs + 1):
    test()
    train(epoch)
    model.update_learning_rate()
    save_checkpoint(epoch)
