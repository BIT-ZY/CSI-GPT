#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import math
import torch
from torch import nn


from options import args_parser
from update import LocalUpdate, update_model_inplace,compute_nmse
from utils import get_model, get_dataset, average_weights, exp_details, average_parameter_delta,OTA
from scipy.io import savemat
from torch.utils.data import Dataset

class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    
    # define paths
#     out_dir_name = args.model + args.dataset + args.optimizer + '_lr' + str(args.lr) + '_locallr' + str(args.local_lr) + '_localep' + str(args.local_ep) +'_localbs' + str(args.local_bs) + '_eps' + str(args.eps)
    file_name = '/{}_{}_{}_llr[{}]_glr[{}]_eps[{}]_le[{}]_bs[{}]_iid[{}]_mi[{}]_frac[{}].pkl'.\
                format(args.dataset, args.model, args.optimizer, 
                    args.local_lr, args.lr, args.eps, 
                    args.local_ep, args.local_bs, args.iid, args.max_init, args.frac)
    # logger = SummaryWriter('./logs/'+file_name)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # limit cpu use
    print ('-- pytorch version: ', torch.__version__)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    # load dataset and user groups
    # train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)
    train_data, test_data, train_user_groups = get_dataset(args)

    test_dataset = DatasetFolder(test_data)
    test_dl_local = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # Set the model to train and send it to device.
    NUM_FEEDBACK_BITS =128
    global_model = get_model(args.model, args.dataset, img_size=None,NUM_FEEDBACK_BITS=NUM_FEEDBACK_BITS, snr=args.snr, cr=args.cr)

    # 加载训练模型
    global_encoder_para = global_model.encoder.state_dict()
    global_decoder_para = global_model.decoder.state_dict()
    pretrained_encoder_dict = torch.load(
        '')# load pretrained model
    pretrained_decoder_dict = torch.load(
        '')# load pretrained model

    pretrained_encoder_dict = {k: v for k, v in pretrained_encoder_dict['state_dict'].items() if
                               k in global_encoder_para}
    pretrained_decoder_dict = {k: v for k, v in pretrained_decoder_dict['state_dict'].items() if
                               k in global_decoder_para}
    global_encoder_para.update(pretrained_encoder_dict)
    global_decoder_para.update(pretrained_decoder_dict)
    global_model.encoder.load_state_dict(global_encoder_para)
    global_model.decoder.load_state_dict(global_decoder_para)



    global_model.to(device)

    global_model.eval()
    test_nmse_dB = compute_nmse(global_model, test_dl_local, args, device=device)
    print('>> Global Model Test avg nmse真实信道与重构: %f' % test_nmse_dB)

    global_model.train()
    
    
    momentum_buffer_list = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = [] 
    for i, p in enumerate(global_model.parameters()):         
        momentum_buffer_list.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avgs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        max_exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False)+args.max_init) # 1e-3
    
    
 
    
    
    # Training
    train_loss_sampled, train_loss, train_accuracy = [], [], []
    test_loss, test_accuracy = [], []
    start_time = time.time()
    loss_history = []
    bestloss = 10
    for epoch in tqdm(range(args.epochs)):
        ep_time = time.time()
        
        local_weights, local_params, local_losses = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        par_before = []
        for p in global_model.parameters():  # get trainable parameters
            par_before.append(p.data.detach().clone())
        # this is to store parameters before update
        w0 = global_model.state_dict()  # get all parameters, includeing batch normalization related ones
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            
            local_model = LocalUpdate(args=args,
                                      tarin_data=train_data,
                                      train_idxs=train_user_groups[idx],
                                      )
            
            w, p, loss = local_model.update_weights_local(
                model=copy.deepcopy(global_model), global_round=epoch) #w 是model.state_dict()是一个字典，对应每一层的名字和参数，最后的loss返回的是所有本地训练中的epoch的均值，
            #W是字典 P是parameter参数
            
            local_weights.append(copy.deepcopy(w))
            local_params.append(copy.deepcopy(p))
            local_losses.append(copy.deepcopy(loss))

        bn_weights = average_weights(local_weights)#进行FedAVg 这里似乎需要考虑用户数据量相同才成立，因为直接是平均
        global_model.load_state_dict(bn_weights)
        
        # this is to update trainable parameters via different optimizers
        global_delta = average_parameter_delta(local_params, par_before) # calculate compression in this function

        # OTA应该在这里加噪对于 global_delta 进行加噪
        global_delta_withnoise = OTA(global_delta,args.snr)

        update_model_inplace(
            global_model, par_before, global_delta_withnoise, args, epoch,
            momentum_buffer_list, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    

        # report and store loss and accuracy
        # this is local training loss on sampled users
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        print('Epoch Run Time: {0:0.4f} of {1} global rounds'.format(time.time()-ep_time, epoch+1))
        print(f'Training Loss : {train_loss[-1]}')
        # logger.add_scalar('train loss', train_loss[-1], epoch)

        global_model.eval()
        
         
        # Test inference after completion of training


        # test_acc, test_ls = test_inference(args, global_model, test_dataset)
        # test_accuracy.append(test_acc)
        # test_loss.append(test_ls)


        test_nmse_dB = compute_nmse(global_model, test_dl_local, args, device=device)
        print('>> Global Model Test avg nmse真实信道与重构: %f' % test_nmse_dB)

        loss_history.append(test_nmse_dB)
        savemat(str(args.optimizer)+str(NUM_FEEDBACK_BITS)+str(args.frac)+ str(args.local_ep)+'_test_loss.mat', {'loss': loss_history})
        savemat(str(args.optimizer)+str(NUM_FEEDBACK_BITS)+str(args.frac)+ str(args.local_ep)+'_train_loss.mat', {'loss': train_loss})
        if test_nmse_dB < bestloss:

            torch.save({'state_dict': global_model.encoder.state_dict(
            ), }, '.\\model_saved/encoder_q_large' + str(NUM_FEEDBACK_BITS) + str(args.snr)+ str(args.optimizer)+ str(args.frac)+ str(args.local_ep)+'.pth.tar')
            # Decoder Saving
            torch.save({'state_dict': global_model.decoder.state_dict(
            ), }, '.\\model_saved/decoder_q_large' + str(NUM_FEEDBACK_BITS) + str(args.snr)+ str(args.optimizer)+ str(args.frac)+ str(args.local_ep)+'.pth.tar')
            print("Model saved")
            bestloss = test_nmse_dB
        # print global training loss after every rounds




                

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

