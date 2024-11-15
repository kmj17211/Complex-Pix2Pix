from os.path import join
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class SAR_Dataset(Dataset):
    def __init__(self, data_type, transform = False, train = True, ep = 1e-6, clip = 10):
        super(SAR_Dataset).__init__()

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()

        path = './Data/SAR Data/SAMPLE'
        self.data_type = data_type
        self.train = train
        self.ep = ep
        self.clip = clip

        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

        if data_type == 'VGG':

            if train:
                self.dir_t = 'gen_complex'
            else:
                self.dir_t = 'real/train'

            dir = 'png_images_pix2pix_60%/qpm'

            self.path2png = join(path, dir, self.dir_t)
            path2folder = self.path2png
            self.getitem = self._vgg_getitem

        else:

            if train:
                self.dir_t = 'train'
                self.label = self.label[:7]
            else:
                self.dir_t = 'test'

            if data_type == 'Complex':
                dir_an = 'mat_files_pix2pix_60%'
                dir_mg = 'png_images_pix2pix_60%/qpm'

                self.path2png = join(path, dir_mg)
                self.path2mat = join(path, dir_an)

                self.getitem = self._complex_getitem

            elif data_type == 'QPM':
                dir_mg = 'png_images_pix2pix_60%/qpm'
                self.path2png = join(path, dir_mg)
                self.getitem = self._png_getitem

            elif data_type == 'Abs':
                dir_mg = 'mat_files_pix2pix_60%'
                self.path2mat = join(path, dir_mg)
                self.getitem = self._abs_getitem

            else:
                raise Exception('Data Type을 Complex나 QPM 둘 중 하나로 입력해')
        
            # path2folder = join(self.path2png, 'real', self.dir_t)
            path2folder = join(self.path2mat, 'real', self.dir_t)
        
        # self.label = ['bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
        
        self.data_name = []
        self.data_label = []
        for label in self.label:
            path2data = join(path2folder, label)
            data_name = os.listdir(path2data)
            self.data_name.extend(data_name)

            self.data_label.extend([label] * len(data_name))


    def _complex_getitem(self, index):
        
        # Label
        label = self.data_label[index]

        # Data Path
        real_mg_data_path = join(self.path2png, 'real', self.dir_t, label, self.data_name[index])
        real_an_data_path = join(self.path2mat, 'real', self.dir_t, label, self.data_name[index][:-3] + 'mat')

        synth_mg_data_path = join(self.path2png, 'synth', self.dir_t, label, self.data_name[index].replace('real', 'synth'))
        synth_an_data_path = join(self.path2mat, 'synth', self.dir_t, label, self.data_name[index].replace('real', 'synth')[:-3] + 'mat')

        # Data Load
        real_mg = self.transform(Image.open(real_mg_data_path).convert('L'))
        real_an = self.transform(loadmat(real_an_data_path)['complex_img']).angle()

        synth_mg = self.transform(Image.open(synth_mg_data_path).convert('L'))
        synth_an = self.transform(loadmat(synth_an_data_path)['complex_img']).angle()

        real_R, real_I = real_mg * torch.cos(real_an), real_mg * torch.sin(real_an)
        synth_R, synth_I = synth_mg * torch.cos(synth_an), synth_mg * torch.sin(synth_an)
        real, synth = torch.complex(real_R, real_I), torch.complex(synth_R, synth_I)

        if self.train:
            return synth.type(torch.complex64), real.type(torch.complex64), label
        else:
            return synth.type(torch.complex64), real.type(torch.complex64), label, self.data_name[index]
    
    def _abs_getitem(self, index):
        
        # Label
        label = self.data_label[index]

        # Data Path
        real_abs_data_path = join(self.path2mat, 'real', self.dir_t, label, self.data_name[index][:-3] + 'mat')
        synth_abs_data_path = join(self.path2mat, 'synth', self.dir_t, label, self.data_name[index].replace('real', 'synth')[:-3] + 'mat')

        # Data Load
        real_abs = abs(loadmat(real_abs_data_path)['complex_img'])
        synth_abs = abs(loadmat(synth_abs_data_path)['complex_img'])
        real_abs = np.log10(real_abs + self.ep)
        synth_abs = np.log10(synth_abs + self.ep)
        real_abs[real_abs > np.log10(self.clip)] = np.log10(self.clip)
        synth_abs[synth_abs > np.log10(self.clip)] = np.log10(self.clip)
        real_abs = (real_abs - np.log10(self.ep)) / (np.log10(self.clip) - np.log10(self.ep))
        synth_abs = (synth_abs - np.log10(self.ep)) / (np.log10(self.clip) - np.log10(self.ep))

        real_abs = self.transform(real_abs)
        synth_abs = self.transform(synth_abs)

        if self.train:
            return synth_abs.type(torch.float32), real_abs.type(torch.float32), label
        else:
            return synth_abs.type(torch.float32), real_abs.type(torch.float32), label, self.data_name[index]

    def _png_getitem(self, index):
        
        # Label
        label = self.data_label[index]

        # Data Path
        real_data_path = join(self.path2png, 'real', self.dir_t, label, self.data_name[index])
        synth_data_path = join(self.path2png, 'synth', self.dir_t, label, self.data_name[index].replace('real', 'synth'))

        # Data Load
        real = self.transform(Image.open(real_data_path).convert('L'))
        synth = self.transform(Image.open(synth_data_path).convert('L'))

        if self.train:
            return synth, real, label
        else:
            return synth, real, label, self.data_name[index]
    
    def _vgg_getitem(self, index):

        # Label
        label_str = self.data_label[index]
        label = torch.zeros(len(self.label))
        label[self.label.index(label_str)] = 1

        # Data Path
        img_path = join(self.path2png, label_str, self.data_name[index])

        # Data Load
        img = self.transform(Image.open(img_path).convert('L'))

        return img, label
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return len(self.data_name)

if __name__ == '__main__':
    ds = SAR_Dataset('Abs', train=True)

    for i, (synth, real,_) in enumerate(ds):
        if i == 0:
            synth_cat = [synth]
            real_cat = [real]
        else:
            synth_cat.append(synth)
            real_cat.append(real)
    synth = torch.cat(synth_cat)
    real = torch.cat(real_cat)

    db = 1
    ep = 1e-6
    clip = 2
    normalize = True

    print('Flatting Synthetic img...')
    # synth_flat = db * np.log10(synth.flatten() + ep)
    # synth_flat[synth_flat > np.log10(clip)] = np.log10(clip)
    synth[synth > clip] = clip
    synth_flat = np.log10(1000 * synth.flatten() + 1) / np.log10(1000 * clip + 1)
    print('Flatting Real img...')
    # real_flat = db * np.log10(real.flatten() + ep)
    # real_flat[real_flat > np.log10(clip)] = np.log10(clip)
    real[real > clip] = clip
    real_flat = np.log10(1000 * real.flatten() + 1) / np.log10(1000 * clip + 1)

    if normalize:
        # synth_min = synth_flat.min()
        # real_min = real_flat.min()
        # synth_flat = synth_flat - synth_min
        # real_flat = real_flat - real_min
        # synth_max = synth_flat.max()
        # real_max = real_flat.max()
        synth_flat = (synth_flat - 0.5) / 0.5
        real_flat = (real_flat - 0.5) / 0.5

    print('Counting a number of Pixel Intensity for Histogram')
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.title('Synthetic (Simul) Mean: ' + str(synth_flat.mean()) + ', Std: ' + str(synth_flat.std()))
    plt.hist(synth_flat, bins = 100, log = False)
    plt.subplot(2, 1, 2)
    plt.title('Measured (Real) Mean: ' + str(real_flat.mean()) + ', Std: ' + str(real_flat.std()))
    plt.hist(real_flat, bins = 100, log = False)
    plt.show()
    plt.savefig('a.png')