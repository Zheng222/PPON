import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image


def np2tensor():
    return Compose([
        ToTensor(),
    ])



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg"])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


class DatasetFromFolderVal(data.Dataset):
    def __init__(self, hr_dir, lr_dir, upscale):
        super(DatasetFromFolderVal, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        input = load_image(self.lr_filenames[index])
        target = load_image(self.hr_filenames[index])
        input = np2tensor()(input)
        target = np2tensor()(target)

        return input, target

    def __len__(self):
        return len(self.lr_filenames)
