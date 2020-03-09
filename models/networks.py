import functools
from torch.nn import init
from . import architecture as arch

#########################
# initialize
#########################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1.0, std=0.02):
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


########################
# define network
########################

def define_G(args):
    if args.which_model == 'content':
        netG = arch.PPON_content(in_nc=3, nf=64, nb=24, out_nc=3)
    elif args.which_model == 'structure':
        netG = arch.PPON_structure(in_nc=3, nf=64, nb=24, out_nc=3)
    elif args.which_model == 'perceptual':
        netG = arch.PPON_perceptual(in_nc=3, nf=64, nb=24, out_nc=3)
    elif args.which_model == 'ppon':
        netG = arch.PPON(in_nc=3, nf=64, nb=24, out_nc=3, alpha=args.alpha)
    else:
        raise NotImplementedError('Generator models [{:s}] not recognized'.format(args.which_model))
    if args.is_train:
        init_weights(netG, init_type='kaiming', scale=0.1)
    return netG

def define_D():
    netD = arch.Discriminator_192(in_nc=3, base_nf=64)
    init_weights(netD, init_type='kaiming', scale=1)
    return netD


def define_F(use_bn=False):
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True)
    netF.eval()
    return netF
