import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]

class VAE(nn.Module):  # 定义VAE模型
    def __init__(self, img_size, latent_dim):  # 初始化方法
        super(VAE, self).__init__()  # 继承初始化方法
        self.in_channel, self.img_h, self.img_w = img_size  # 由输入图片形状得到图片通道数C、图片高度H、图片宽度W
        self.h = 1  # 经过5次卷积后，最终特征层高度变为原图片高度的1/32
        self.w = 1  # 经过5次卷积后，最终特征层宽度变为原图片高度的1/32
        hw = self.h * self.w  # 最终特征层的尺寸hxw
        self.latent_dim = latent_dim  # 采样变量Z的长度
        self.hidden_dims_input = [64, 128, 256, 512]  # 特征层通道数列表
        self.hidden_dims = [32,64, 128, 256, 512]  # 特征层通道数列表
        # 开始构建编码器Encoder
        layers = []  # 用于存放模型结构
        layers += [nn.Conv2d(self.in_channel, 32, 32, 16, 8),  # 添加conv
                   nn.BatchNorm2d(32),  # 添加bn
                   nn.LeakyReLU()]  # 添加leakyrelu
        self.in_channel = 32  # 将下次循环的输入通道数设为本次循环的输出通道数
        for hidden_dim in self.hidden_dims_input:  # 循环特征层通道数列表
            layers += [nn.Conv2d(self.in_channel, hidden_dim, 3, 2, 1),  # 添加conv
                       nn.BatchNorm2d(hidden_dim),  # 添加bn
                       nn.LeakyReLU()]  # 添加leakyrelu
            self.in_channel = hidden_dim  # 将下次循环的输入通道数设为本次循环的输出通道数

        self.encoder = nn.Sequential(*layers)  # 解码器Encoder模型结构

        self.fc_mu = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linaer，将特征向量转化为分布均值mu
        self.fc_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linear，将特征向量转化为分布方差的对数log(var)
        # 开始构建解码器Decoder
        layers = []  # 用于存放模型结构
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)  # linaer，将采样变量Z转化为特征向量
        self.hidden_dims.reverse()  # 倒序特征层通道数列表
        for i in range(len(self.hidden_dims) - 1):  # 循环特征层通道数列表
            layers += [nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1], 3, 2, 1, 1),  # 添加transconv
                       nn.BatchNorm2d(self.hidden_dims[i + 1]),  # 添加bn
                       nn.LeakyReLU()]  # 添加leakyrelu
        layers += [nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1], 32, 16, 8, 0),  # 添加transconv
                   nn.BatchNorm2d(self.hidden_dims[-1]),  # 添加bn
                   nn.LeakyReLU(),  # 添加leakyrelu
                   nn.Conv2d(self.hidden_dims[-1], img_size[0], 3, 1, 1),  # 添加conv
                   ]  # 添加tanh
        self.decoder = nn.Sequential(*layers)  # 编码器Decoder模型结构

    def encode(self, x):  # 定义编码过程
        result = self.encoder(x)  # Encoder结构,(n,1,32,32)-->(n,512,1,1)
        result = torch.flatten(result, 1)  # 将特征层转化为特征向量,(n,512,1,1)-->(n,512)
        mu = self.fc_mu(result)  # 计算分布均值mu,(n,512)-->(n,128)
        log_var = self.fc_var(result)  # 计算分布方差的对数log(var),(n,512)-->(n,128)
        return [mu, log_var]  # 返回分布的均值和方差对数

    def decode(self, z):  # 定义解码过程
        y = self.decoder_input(z).view(-1, self.hidden_dims[0], self.h,
                                       self.w)  # 将采样变量Z转化为特征向量，再转化为特征层,(n,128)-->(n,512)-->(n,512,1,1)
        y = self.decoder(y)  # decoder结构,(n,512,1,1)-->(n,1,32,32)
        return y  # 返回生成样本Y

    def reparameterize(self, mu, log_var):  # 重参数技巧
        std = torch.exp(0.5 * log_var)  # 分布标准差std
        eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
        return mu + eps * std  # 返回对应正态分布中的采样值

    def forward(self, x):  # 前传函数
        mu, log_var = self.encode(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
        z = self.reparameterize(mu, log_var)  # 经过重参数技巧，得到分布采样变量Z
        y = self.decode(z)  # 经过解码过程，得到生成样本Y
        return [y, x, mu, log_var]  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var

    def sample(self, n, cuda):  # 定义生成过程
        z = torch.randn(n, self.latent_dim)  # 从标准正态分布中采样得到n个采样变量Z，长度为latent_dim
        if cuda:  # 如果使用cuda
            z = z.cuda()  # 将采样变量Z加载到GPU
        images = self.decode(z)  # 经过解码过程，得到生成样本Y
        return images  # 返回生成样本Y


def loss_fn(y, x, mu, log_var):  # 定义损失函数
    recons_loss = F.mse_loss(y, x)  # 重建损失，MSE
    kld_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)  # 分布损失，正态分布与标准正态分布的KL散度
    return recons_loss + w * kld_loss  # 最终损失由两部分组成，其中分布损失需要乘上一个系数w


if __name__ == "__main__":
    total_epochs = 1000  # epochs
    batch_size = 64  # batch size
    lr = 5e-4  # lr
    w = 0.00025  # kld_loss的系数w
    num_workers = 8  # 数据加载线程数
    image_size = 256  # 图片尺寸
    image_channel = 2  # 图片通道
    latent_dim = 512  # 采样变量Z长度
    sample_images_dir = ""  # 生成样本示例存放路径


    os.makedirs(sample_images_dir, exist_ok=True)  # 创建生成样本示例存放路径

    cuda = True if torch.cuda.is_available() else False  # 如果cuda可用，则使用cuda
    img_size = (image_channel, image_size, image_size)  # 输入样本形状(1,32,32)

    vae = VAE(img_size, latent_dim)  # 实例化VAE模型，传入输入样本形状与采样变量长度
    if cuda:  # 如果使用cuda
        vae = vae.cuda()  # 将模型加载到GPU
    # dataset and dataloader
    train = ''#load dataset
    data_train = np.load(train)
    data_train = np.transpose(data_train, [0, 3, 1, 2])
    data_train = data_train.astype('float32')  # 训练变量类型转换

    train_dataset = DatasetFolder(data_train)  # [数据，标签] = [下行，上行]
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # test 环节



    # optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)  # 使用Adam优化器
    # train loop

    for epoch in range(total_epochs):  # 循环epoch
        total_loss = 0  # 记录总损失
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{total_epochs}", postfix=dict,
                    miniters=0.3)  # 设置当前epoch显示进度
        for i, img in enumerate(train_loader):  # 循环iter
            if cuda:  # 如果使用cuda
                img = img.cuda()  # 将训练数据加载到GPU
            vae.train()  # 模型开始训练
            optimizer.zero_grad()  # 模型清零梯度
            y, x, mu, log_var = vae(img)  # 输入训练样本X，得到生成样本Y，输入样本X，分布均值mu，分布方差对数log_var
            loss = loss_fn(y, x, mu, log_var)  # 计算loss
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度，更新网络参数
            total_loss += loss.item()  # 累计loss
            pbar.set_postfix(**{"Loss": loss.item()})  # 显示当前iter的loss
            pbar.update(1)  # 步进长度
        pbar.close()  # 关闭当前epoch显示进度
        print("total_loss:%.4f" %
              (total_loss / len(train_loader)))  # 显示当前epoch训练完成后，模型的总损失
        if epoch %50 == 0:
            vae.eval()  # 模型开始验证
            sample_images = vae.sample(25, cuda)  # 获得25个生成样本
            test_images = sample_images
            test_images = test_images[:2, :, :, :].view(-1, 2, 256, 256)  # [1,2,256,256]
            test_images_complex = test_images[:, 0, :, :] + 1j * test_images[:, 1, :, :]

            fig, ax = plt.subplots()
            ax.imshow(np.abs(test_images_complex[0].cpu().data.numpy()), cmap='jet')
            ax.set_title('Generate downlink channel!')

            label = 'Epoch {0}'.format(epoch)
            fig.text(0.5, 0.04, label, ha='center')
            # plt.show()

            plt.savefig('./'+sample_images_dir+'/VAE_result'+str(epoch+1)+'.png')
            torch.save(vae.state_dict(), './'+sample_images_dir +'/'+ str(epoch + 1) + 'vae_param.pkl')