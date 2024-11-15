import matplotlib.pyplot as plt
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision import transforms

from dataset_supervised import SAR_Dataset
from network.pix2pix import Resnet_Generator, Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

# def initialize_weights(model):
#     class_name = model.__class__.__name__
#     if class_name.find('Complex') != -1:
#         pass
#     elif class_name.find('Conv2d') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.02)

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    ep = 1e-3
    clip = 2

    # 데이터셋 불러오기
    train_ds = SAR_Dataset('Abs', train = True, transform = transform, ep = ep, clip = clip)
    path = './Weight/pix2pix'

    # Tensorboard
    writer = SummaryWriter(path + '/run')

    # 샘플 이미지 확인하기
    a,b,t = train_ds[100]
    plt.subplot(1,2,1)
    plt.imshow(a.squeeze() * 0.5 + 0.5, cmap = 'gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(b.squeeze() * 0.5 + 0.5, cmap = 'gray')
    plt.axis('off')
    plt.title(t)
    plt.show()
    plt.savefig('aa.png')
    
    # 데이터 로더 생성하기
    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model_gen = Resnet_Generator().to(device)
    model_dis = Discriminator().to(device)

    # 가중치 초기화 적용
    model_gen.apply(initialize_weights)
    model_dis.apply(initialize_weights)

    # 손실함수 (Non Saturation -> Least Square)
    # loss_func_gan = nn.BCELoss()
    loss_func_gan = nn.MSELoss()
    loss_func_pix = nn.L1Loss()

    # loss_func_pix 가중치
    lambda_pixel = 100

    # patch 수
    patch = (1,15,15)

    # 최적화 파라미터
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999

    opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))
    sche_dis = optim.lr_scheduler.StepLR(opt_dis, step_size = 50, gamma = 0.5)
    sche_gen = optim.lr_scheduler.StepLR(opt_gen, step_size = 50, gamma = 0.5)

    # 학습
    model_gen.train()
    model_dis.train()

    num_epochs = 300
    start_time = time.time()

    print('Train Start')
    batch_count = 0
    for epoch in range(num_epochs):
    
        for a, b, _ in train_dl: # a : input, b : wonder image
            ba_si = a.size(0)

            # real image
            real_a = a.to(device)
            real_b = b.to(device)

            # patch label
            real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

            # discriminator
            model_dis.zero_grad()

            fake_b = model_gen(real_a) # 가짜 이미지 생성

            out_dis = model_dis(real_b, real_a) # 진짜 이미지 식별
            real_loss = loss_func_gan(out_dis,real_label)
            
            out_dis = model_dis(fake_b.detach(), real_a) # 가짜 이미지 식별
            fake_loss = loss_func_gan(out_dis,fake_label)

            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            opt_dis.step()

            # generator
            model_gen.zero_grad()

            out_dis = model_dis(fake_b, real_a) # 가짜 이미지 식별

            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_b, real_b)

            g_loss = gen_loss + lambda_pixel * pixel_loss

            g_loss.backward()
            opt_gen.step()

            # Tensorboard
            writer.add_scalar("G_Loss/train", g_loss, batch_count)
            writer.add_scalar("D_Loss/train", d_loss, batch_count)

            batch_count += 1

        print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
        plot_fake_b = make_grid(fake_b * 0.5 + 0.5, nrow = 16, padding = 20, pad_value = 0.5)
        writer.add_image("Generated_Image", plot_fake_b, batch_count)

        sche_dis.step()
        sche_gen.step()

    writer.close()

    # 가중치 저장
    path2models = path
    os.makedirs(path2models, exist_ok=True)
    path2weights_gen = os.path.join(path2models, 'weights_gen_230827_no_aug1.pt')
    # path2weights_dis = os.path.join(path2models, 'weights_dis1.pt')


    torch.save(model_gen.state_dict(), path2weights_gen)
    # torch.save(model_dis.state_dict(), path2weights_dis)

    # # 가중치 불러오기
    # weights = torch.load(path2weights_gen)
    # model_gen.load_state_dict(weights)

    # # evaluation model
    # model_gen.eval()

    # # 가짜 이미지 생성
    # with torch.no_grad():
    #     for a,b,_ in train_dl:
    #         fake_imgs = model_gen(a.to(device)).detach().cpu()
    #         real_imgs = b
    #         break

    # # 가짜 이미지 시각화
    # plt.figure(figsize=(10,10))

    # for ii in range(0,16,2):
    #     plt.subplot(4,4,ii+1)
    #     plt.imshow(real_imgs[ii].squeeze())
    #     plt.title('Real Image')
    #     plt.axis('off')
    #     plt.subplot(4,4,ii+2)
    #     plt.imshow(fake_imgs[ii].squeeze())
    #     plt.title('Synthetic Image')
    #     plt.axis('off')
    # plt.show()
