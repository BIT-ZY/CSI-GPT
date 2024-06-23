#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import channel_fixed_iid
import h5py

import numpy as np
from SWT_final import  SWTCAN
# from models.randaug import RandAugment

def get_model(model_name, dataset, img_size, NUM_FEEDBACK_BITS, snr, cr):
    if model_name == 'vggnet':
        from models import vgg
        model = vgg.VGG('VGG11', num_classes=nclass)
        
    elif model_name == 'resnet':
        from models import resnet
        model = resnet.ResNet18(num_classes=nclass)
        
    elif model_name == 'wideresnet':
        from models import wideresnet
        model = wideresnet.WResNet_cifar10(num_classes=nclass, depth=16, multiplier=4)
        
    elif model_name == 'cnnlarge':
        from models import simple
        model = simple.CNNLarge()
        
    elif model_name == 'convmixer':
        from models import convmixer
        model = convmixer.ConvMixer(n_classes=nclass)
    
    elif model_name == 'cnn':
        from models import simple
        
        if dataset == 'mnist':
            model = simple.CNNMnist(num_classes=nclass, num_channels=1)
        elif dataset == 'fmnist':
            model = simple.CNNFashion_Mnist(num_classes=nclass)
        elif dataset == 'cifar':
            model = simple.CNNCifar(num_classes=nclass)
    elif model_name == 'ae':
        from models import simple
        
        if dataset == 'mnist' or dataset == 'fmnist':
            model = simple.Autoencoder()
         
    elif model_name == 'mlp':
        from models import simple

        len_in = 1
        for x in img_size:
            len_in *= x
            model = simple.MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=nclass)

    elif model_name == 'SWTCAN':
        print(snr)
        print(NUM_FEEDBACK_BITS)
        model = SWTCAN(num_feedbit=NUM_FEEDBACK_BITS,snr=snr,cr=cr)
    else:
        exit('Error: unrecognized model')

    return model


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """




    if args.dataset == 'CDL':
        print(f'   CDL:\n')
        # dl_obj = channel_truncated
        train = 'D:\zy\CDLv2\H_train_down_B_6kv2_41.npy'
        test = 'D:\zy\CDLv2\H_test_down_B_2kv2_41.npy'

        data_train = np.load(train)

        data_train = data_train.astype('float32')  # 训练变量类型转换

        data_test = np.load(test)
        data_test = data_test.astype('float32')  # 训练变量类型转换
        train_user_groups = channel_fixed_iid(data_train, args.num_users)
        # test_user_groups = channel_fixed_iid(data_test, args.num_users)
    else:
        exit('fuck')

    # return data_train, data_test, train_user_groups, test_user_groups
    return data_train, data_test, train_user_groups



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_parameter_delta(ws, w0):
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            w_avg[key] += ws[i][key] - w0[key]
        w_avg[key] = torch.div(w_avg[key], len(ws))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    dataset   : {args.dataset}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z


def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z


def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z

def OTA(list,snr):
    list_nois = []
    for i in range(len(list)):
        signal = list[i]
        signal_power = signal**2

        noise_var = signal_power / (10 ** (snr / 10))
        noise = torch.randn(size=signal.shape).cuda()
        noise = torch.sqrt(noise_var).cuda() * noise
        signal = signal+noise
        list_nois.append(signal)
    return list_nois