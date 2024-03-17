import argparse
import os
import sys
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from myImageDataset import myImageDataset
from unet import AdUNet
from unet import *

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as mse

from math import sqrt


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
# 轮次修改，本地跑太慢，需要借用一下别的电脑
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0008, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()


# Initialize net
net = AdUNet()
# Losses
loss = torch.nn.L1Loss()

# 数据传给显卡？
cuda = torch.cuda.is_available()
if cuda:
    net = net.cuda()
    gloss = loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    net.load_state_dict(torch.load("./saved_models/AdUnet_%d.pkl"))

# Optimizers
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_loader = DataLoader(
    dataset=myImageDataset(inputs_root=r"./512/origin",
                         labels_root=r"./512/label"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu
)
test_loader = DataLoader(
    dataset=myImageDataset(inputs_root=r"./512/testOrigin",
                         labels_root=r"./512/testLabel"),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu
)


# ----------
#  Training
# ----------
def train(epoch):
    for i, data in enumerate(train_loader):
        # 一个batch
        inputs, labels = data
        inputs = inputs.unsqueeze(1)
        labels = labels.unsqueeze(1)
        # 将这些数据转换成Variable类型
        inputs, labels = Variable(inputs), Variable(labels)
        device = torch.device("cuda" if cuda else "cpu")
        inputs = inputs.to(device)
        labels = labels.to(device)
        # ------------------
        #  Train net
        # ------------------

        optimizer.zero_grad()  # 梯度归零：step之前要进行梯度归零
        net.train()
        netout = net(inputs)

        # Total loss
        gloss = loss(netout, labels)

        gloss.backward()  # 进行反向传播求出每个参数的梯度
        optimizer.step()  # 更新学习率
##
        # --------------
        #  Log Progress
        # --------------

        # getting the current time
        time_object = time.localtime()

        # format the time
        current_time = time.strftime("%H:%M:%S", time_object)

        # print the time

        print("Current time is: ", current_time)
        
        sys.stdout.write(
            "\n[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_loader), gloss.item())
        )
        with open(r"./data/pigs/loss.txt", "a") as f1:
            f1.write("\n[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                     % (epoch, opt.n_epochs, i, len(train_loader), gloss.item()))
            f1.close()


def precision(orign, result):
    matches = orign == result
    
    # 然后，计算布尔数组中True的比例，即分类正确的像素占总像素的比例
    accuracy = np.mean(matches)
    
    print(f'分类准确度: {accuracy:.4f}')

def test():
    total_s = 0  # ssim
    total_p = 0  # psnr
    total_r = 0  # rmse

    with torch.no_grad():
        tt=0
        for inputs, labels in test_loader:
            # 一个batch
            inputs = inputs.unsqueeze(1)
            # labels = labels.unsqueeze(0)
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs), Variable(labels)
            device = torch.device("cuda" if cuda else "cpu")
            inputs = inputs.to(device)
            labels = labels.squeeze(0).cpu().numpy()

            # optimizer.zero_grad()
            net.eval()
            netout = net(inputs)
            img_out = netout.squeeze(1)
            img_out = img_out.squeeze(0)
            img_out = img_out.cpu().numpy()

            
            
            #每执行一轮都会来这里test一次，所以在这里计算准确度
            out_data = img_out
            out_data =  5 * (out_data - out_data.min()) / (out_data.max() - out_data.min())
            # res = out_data.astype(np.uint8)  tmd这是截断啊啊啊啊啊啊
            res = np.round(out_data).astype(np.uint8)
            precision(res,labels)

            np.savetxt("./resrecord/"+str(epoch)+"-"+str(tt) + ".txt", res)
            tt+=1

            
            
            # 展示0-6的数值数量
            # 使用numpy.bincount计算每个值的出现次数
            counts = np.bincount(res.flatten())

            print('unet的结果')

            # 打印结果
            for t, count in enumerate(counts):
                print(f"值 {t} 出现的次数: {count}")
            
            # Total loss
            # gloss = loss(netout, labels)
            # print(netout.shape)
            s = compare_ssim(labels, img_out)
            p = compare_psnr(labels, img_out)
            r = sqrt(mse(labels, img_out))

            total_s += float(s.item())
            total_p += float(p.item())
            total_r += float(r)
    print('\n|Epoch %d/%d| |Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f| '
          % ((epoch + 1), opt.n_epochs, total_s / len(test_loader), total_p / len(test_loader),
             total_r / len(test_loader)))

    with open(r"./data/pigs/test_parameters.txt", "a") as f:
        f.write("|Epoch %d/%d| |Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f| \r\n"
                % ((epoch + 1), opt.n_epochs, total_s / len(test_loader), total_p / len(test_loader),
                   total_r / len(test_loader)))
        f.close()
    return total_s / len(test_loader), total_p / len(test_loader), total_r / len(test_loader)


def test2():
    total_s = 0  # ssim
    total_p = 0  # psnr
    total_r = 0  # rmse
    
    with torch.no_grad():
        t = 0
        for inputs, labels in test_loader:
            # 一个batch
            inputs = inputs.unsqueeze(1)
            # labels = labels.unsqueeze(0)
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs), Variable(labels)
            device = torch.device("cuda" if cuda else "cpu")
            inputs = inputs.to(device)
            labels = labels.squeeze(0).cpu().numpy()
            
            # optimizer.zero_grad()
            net.eval()
            netout = net(inputs)
            img_out = netout.squeeze(1)
            img_out = img_out.squeeze(0)
            img_out = img_out.cpu().numpy()
            

            # 每执行一轮都会来这里test一次，所以在这里计算准确度
            out_data = img_out
            out_data = 5 * (out_data - out_data.min()) / (out_data.max() - out_data.min())
            # res = out_data.astype(np.uint8)  tmd这是截断啊啊啊啊啊啊
            res = np.round(out_data).astype(np.uint8)
            precision(res, labels)

            print("结果输出，labels的shape：")
            print(res.shape)
            np.savetxt(str(t) + ".txt", res)
            t += 1
            
            # 展示0-6的数值数量
            # 使用numpy.bincount计算每个值的出现次数
            counts = np.bincount(res.flatten())
            
            print('unet的结果')
            
            # 打印结果
            for i, count in enumerate(counts):
                print(f"值 {i} 出现的次数: {count}")
            
            # Total loss
            # gloss = loss(netout, labels)
            # print(netout.shape)
            s = compare_ssim(labels, img_out)
            p = compare_psnr(labels, img_out)
            r = sqrt(mse(labels, img_out))
            
            total_s += float(s.item())
            total_p += float(p.item())
            total_r += float(r)
    print('\n|Epoch %d/%d| |Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f| '
          % ((epoch + 1), opt.n_epochs, total_s / len(test_loader), total_p / len(test_loader),
             total_r / len(test_loader)))
    
    with open(r"./data/pigs/test_parameters.txt", "a") as f:
        f.write("|Epoch %d/%d| |Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f| \r\n"
                % ((epoch + 1), opt.n_epochs, total_s / len(test_loader), total_p / len(test_loader),
                   total_r / len(test_loader)))
        f.close()
    return total_s / len(test_loader), total_p / len(test_loader), total_r / len(test_loader)

if __name__ == '__main__':
    for epoch in range(opt.epoch, opt.n_epochs):
        # 一个epoch
        train(epoch)
        # Save model checkpoints
        # torch.save(net.state_dict(), "saved_models/generator_%d.pkl" % epoch)
        torch.save(net, "./saved_models/AdUnet_%d.pkl" % epoch)
        epoch_s, epoch_p, epoch_r = test()

    test2()
