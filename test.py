import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os
from torchvision.transforms import ToPILImage

from dataset_supervised import SAR_Dataset
from network.pix2pix import Resnet_Generator

def to_image(img, ii, title, img2 = None):
    img = np.squeeze(img.numpy())
    if img2 != None:
        img2 = np.squeeze(img2.numpy())
        img = img - img2

    plot(img, ii, title)
    # if Sub == True:
    #     plt.clim(0, 1)
    # else:
    #     plt.clim(-1, 1)
    # plot(mag, ii+5, title+' Mag')
    # plot(phase, ii+10, title+' Phase')

def plot(img, ii, title):
    plt.subplot(6, 5, ii)
    plt.imshow(img, cmap = 'gray')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    ep = 1e-3
    clip = 2

    mu = 0.5
    gamma = 0.5

    train_ds = SAR_Dataset('Abs', train = False, transform = transform, ep = ep, clip = clip)
    test_dl = DataLoader(train_ds, batch_size=8, shuffle = False)

    model_gen = Resnet_Generator().to(device)

    path2model = './Weight/pix2pix/weights_gen_230827_no_aug1.pt'
    path2save = './Data/SAR Data/SAMPLE/results/230822~/pix2pix_230827_no_aug'
    weights = torch.load(path2model)
    model_gen.load_state_dict(weights)
    model_gen.eval()

    to_pil = ToPILImage()

    with torch.no_grad():
        for synth, real, label, name in test_dl:
            fake_real = model_gen(synth.to(device)).detach().cpu()

            for ii, img in enumerate(fake_real):
                path = os.path.join(path2save, 'refine', label[ii])
                os.makedirs(path, exist_ok = True)
                img = to_pil(img * mu + gamma)
                img.save(os.path.join(path, name[ii][:-3]+'png'))
            
            for ii, img in enumerate(synth):
                path = os.path.join(path2save, 'synth', label[ii])
                os.makedirs(path, exist_ok = True)
                img = to_pil(img * mu + gamma)
                img.save(os.path.join(path, name[ii][:-3]+'png'))

            for ii, img in enumerate(real):
                path = os.path.join(path2save, 'real', label[ii])
                os.makedirs(path, exist_ok = True)
                img = to_pil(img * mu + gamma)
                img.save(os.path.join(path, name[ii][:-3]+'png'))

    # plt.hist(real, bins=500, alpha = 0.5, log = True, label = 'Real')
    # plt.hist(synth, bins=500, alpha = 0.5, log = True, label = 'Synth')
    # plt.legend()
    # plt.show()

    # for ii in range(0,30,5):
    #     iii = ii
    #     to_image(real_imgs[iii], ii+1, 'GT')
    #     output_ssim = 'Output ({:.2})'.format(ssim(real_imgs[iii].unsqueeze_(dim=0), fake_imgs[iii].unsqueeze_(dim=0)))
    #     to_image(fake_imgs[iii], ii+2, output_ssim)
    #     input_ssim = 'Input(CAD) ({:.2})'.format(ssim(real_imgs[iii].unsqueeze_(dim=0), synth_imgs[iii].unsqueeze_(dim=0)))
    #     to_image(synth_imgs[iii], ii+3, input_ssim)
    #     to_image(real_imgs[iii], ii+4, 'GT - Output', fake_imgs[iii])
    #     to_image(real_imgs[iii], ii+5, 'GT - Input', synth_imgs[iii])
    # plt.show()