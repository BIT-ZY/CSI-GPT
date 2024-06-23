#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import compressors
import sys
from collections import OrderedDict, defaultdict

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, tarin_data, train_idxs,  logger=None):
        self.args = args
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            tarin_data, list(train_idxs))
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
        # Default criterion set to NLL loss function
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = NMSELoss(reduction='mean').to(self.device)
        ###### define compressors #######
        self.compressor = compressors.Compressor()
        if args.compressor == 'identical':
            self.compressor.makeIdenticalCompressor() 
        elif args.compressor == 'topk256':
            self.compressor.makeTopKCompressor(1/256) 
        elif args.compressor == 'topk128':
            self.compressor.makeTopKCompressor(1/128) 
        elif args.compressor == 'topk64':
            self.compressor.makeTopKCompressor(1/64)   
        elif args.compressor == 'sign':
            self.compressor.makeSignCompressor()
        else:
            exit('unknown compressor: {}'.format(args.compressor))
        


    def train_val_test(self, tarin_data, train_idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        # dl_obj = channel_truncated
        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)

        dl_obj = channel_truncated
        train_dataset = dl_obj(matData=tarin_data, dataidxs=train_idxs, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0, pin_memory=True)
        # test_dataset = dl_obj(matData=test_data, dataidxs=test_idxs, train=True)
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=self.args.local_bs, shuffle=False, num_workers=0, pin_memory=True)

        return train_loader,None, None

    def update_weights_local(self, model, global_round):
        # Set mode to train model
        encoder_need_change_list =[]
        # encoder_need_change_list = ['layers.3.blocks.0.norm1.weight', 'layers.3.blocks.0.norm1.bias',
        #                             'layers.3.blocks.0.attn.relative_position_bias_table',
        #                             'layers.3.blocks.0.attn.qkv.weight', 'layers.3.blocks.0.attn.qkv.bias',
        #                             'layers.3.blocks.0.attn.proj.weight',
        #                             'layers.3.blocks.0.attn.proj.bias', 'layers.3.blocks.0.norm2.weight',
        #                             'layers.3.blocks.0.norm2.bias',
        #                             'layers.3.blocks.1.norm1.weight', 'layers.3.blocks.1.norm1.bias',
        #                             'layers.3.blocks.1.attn.relative_position_bias_table',
        #                             'layers.3.blocks.1.attn.qkv.weight', 'layers.3.blocks.1.attn.qkv.bias',
        #                             'layers.3.blocks.1.attn.proj.weight',
        #                             'layers.3.blocks.1.attn.proj.bias', 'layers.3.blocks.1.norm2.weight',
        #                             'layers.3.blocks.1.norm2.bias',
        #                             ]
        decoder_need_change_list = ['layers_up.2.blocks.0.norm1.weight', 'layers_up.2.blocks.0.norm1.bias',
                                    'layers_up.2.blocks.0.attn.relative_position_bias_table',
                                    'layers_up.2.blocks.0.attn.qkv.weight', 'layers_up.2.blocks.0.attn.qkv.bias',
                                    'layers_up.2.blocks.0.attn.proj.weight',
                                    'layers_up.2.blocks.0.attn.proj.bias', 'layers_up.2.blocks.0.norm2.weight',
                                    'layers_up.2.blocks.0.norm2.bias',
                                    'layers_up.2.blocks.1.norm1.weight', 'layers_up.2.blocks.1.norm1.bias',
                                    'layers_up.2.blocks.1.attn.relative_position_bias_table',
                                    'layers_up.2.blocks.1.attn.qkv.weight', 'layers_up.2.blocks.1.attn.qkv.bias',
                                    'layers_up.2.blocks.1.attn.proj.weight',
                                    'layers_up.2.blocks.1.attn.proj.bias', 'layers_up.2.blocks.1.norm2.weight',
                                    'layers_up.2.blocks.1.norm2.bias',
                                    'layers_up.2.blocks.1.mlp.fc1.weight', 'layers_up.2.blocks.1.mlp.fc1.bias',
                                    'layers_up.2.blocks.1.mlp.fc2.weight', 'layers_up.2.blocks.1.mlp.fc2.bias',
                                    'layers_up.2.upsample.expand.weight', 'layers_up.2.upsample.norm.weight',
                                    'layers_up.2.upsample.norm.bias',

                                    'layers_up.3.blocks.0.norm1.weight', 'layers_up.3.blocks.0.norm1.bias',
                                    'layers_up.3.blocks.0.at tn.relative_position_bias_table',
                                    'layers_up.3.blocks.0.attn.qkv.weight', 'layers_up.3.blocks.0.attn.qkv.bias',
                                    'layers_up.3.blocks.0.attn.proj.weight',
                                    'layers_up.3.blocks.0.attn.proj.bias', 'layers_up.3.blocks.0.norm2.weight',
                                    'layers_up.3.blocks.0.norm2.bias',
                                    'layers_up.3.blocks.1.norm1.weight', 'layers_up.3.blocks.1.norm1.bias',
                                    'layers_up.3.blocks.1.attn.relative_position_bias_table',
                                    'layers_up.3.blocks.1.attn.qkv.weight', 'layers_up.3.blocks.1.attn.qkv.bias',
                                    'layers_up.3.blocks.1.attn.proj.weight',
                                    'layers_up.3.blocks.1.attn.proj.bias', 'layers_up.3.blocks.1.norm2.weight',
                                    'layers_up.3.blocks.1.norm2.bias',
                                    'layers_up.3.blocks.1.mlp.fc1.weight', 'layers_up.3.blocks.1.mlp.fc1.bias',
                                    'layers_up.3.blocks.1.mlp.fc2.weight', 'layers_up.3.blocks.1.mlp.fc2.bias',
                                    'norm_up.weight', 'norm_up.bias', 'up.expand.weight', 'up.norm.weight',
                                    'up.norm.bias'
                                    ]
        # 固定部分参数不动
        for param in model.encoder.named_parameters():
            if param[0] in encoder_need_change_list:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False

        for param in model.decoder.named_parameters():
            if param[0] in decoder_need_change_list:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False
        params = list(model.named_parameters())
        unique_params = {}
        for name, param in params:
            if param.requires_grad and name not in unique_params:
                unique_params[name] = param.numel()
        num_params = sum(unique_params.values())
        print(num_params)

        model.train()
        epoch_loss = []

        # optimizer = NoamOpt(256, 1, 4000, torch.optim.Adam(
        #         [p for p in model.parameters() if p.requires_grad], lr=0, betas=(0.9, 0.98), eps=1e-9))
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.local_lr, momentum=0)

        for iter in range(self.args.local_ep):
            batch_loss = []
            total = 0
            for batch_idx, autoencoderInput in enumerate(self.trainloader):
                autoencoderInput = autoencoderInput.to(self.device)
                # autoencoderInput_noise = get_noise(autoencoderInput,self.args.snr)
                model.zero_grad()
                optimizer.zero_grad()
                out = model(autoencoderInput)
                loss = self.criterion(autoencoderInput, out)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round, iter, batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                
                batch_loss.append(loss.item() * len(autoencoderInput))
                total += len(autoencoderInput)
            epoch_loss.append(sum(batch_loss)/total) #记录每一个epoch的平均loss，由每个epoch中的batchloss构成

        par_after = []
        for p in model.parameters():
            par_after.append(p.data.detach().clone())
            
        
        return model.state_dict(), par_after, sum(epoch_loss) / len(epoch_loss)
     
    
    def compressSignal(self, signal, D):
#         transit_bits = 0
        signal_compressed = []
        for p in signal:
            signal_compressed.append(torch.zeros_like(p))

        signal_flatten = torch.zeros(D).to(self.device)

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_flatten[(signal_offset):(signal_offset + offset)] = signal[t].flatten(0)
            signal_offset += offset
            

        signal_flatten = self.compressor.compressVector(signal_flatten)
#         transit_bits += compressors.Compressor.last_need_to_send_advance

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_compressed[t].flatten(0)[:] = signal_flatten[(signal_offset):(signal_offset + offset)]
            signal_offset += offset

        return signal_compressed 

    def compressSignal_layerwise(self, signal, D):
        transit_bits = 0
#         signal_compressed = []
        for p in signal:
            signal_compressed.append(torch.zeros_like(p))
      
        signal_flatten = torch.zeros(D).to(self.device)

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_flatten[(signal_offset):(signal_offset + offset)] = self.compressor.compressVector(signal[t].flatten(0), self.iteration)
#             transit_bits += compressors.Compressor.last_need_to_send_advance
            signal_offset += offset

        signal_offset = 0
        for t in range(len(signal)):
            offset = len(signal[t].flatten(0))
            signal_compressed[t].flatten(0)[:] = signal_flatten[(signal_offset):(signal_offset + offset)]
            signal_offset += offset

        return signal_compressed 
    

    
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.local_lr, momentum=0)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
#                 self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item()/len(labels))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        par_after = []
        for p in model.parameters():
            par_after.append(p.data.detach().clone())
        
        return par_after, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item() * len(labels)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        loss = loss/total
        return accuracy, loss


def update_model_inplace(model, par_before, delta, args, cur_iter, momentum_buffer_list, exp_avgs, exp_avg_sqs, max_exp_avg_sqs):
    grads = copy.deepcopy(delta)
    
    # learning rate decay
    iteration = cur_iter + 1  # add 1 is to make sure nonzero denominator in adam calculation
    # if iteration < int(args.epochs/2):
    #     lr_decay = 1.0
    # elif iteration < int(3*args.epochs/4):
    #     lr_decay = 0.1
    # else:
    #     lr_decay = 0.01
    lr_decay=1.0

    for i, param in enumerate(model.parameters()): 
        grad = grads[i]  # recieve the aggregated (averaged) gradient
        
        # SGD calculation
        if args.optimizer == 'fedavg':
            # need to reset the trainable parameter
            # because we have updated the model via state_dict when dealing with batch normalization
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).add_(grad, alpha=args.lr * lr_decay)
            # param.data.add_(grad, alpha=args.lr * lr_decay)
        # SGD+momentum calculation
        elif args.optimizer == 'fedavgm':
            buf = momentum_buffer_list[i]
            buf.mul_(args.momentum).add_(grad, alpha=1)
            grad = buf
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).add_(grad, alpha=args.lr * lr_decay)
        # adam calculation
        elif args.optimizer == 'fedadam':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(args.eps) # without maximum

            step_size = args.lr * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedams':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(args.eps)

            step_size = args.lr * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedamsd':
            lr_decay=1.0/math.sqrt(iteration)
            
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(args.eps)

            step_size = args.lr * lr_decay / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        elif args.optimizer == 'fedadagrad':
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_sq.addcmul_(1, grad, grad)            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(grad, exp_avg_sq.sqrt().add_(args.eps), value=args.lr * lr_decay)
        elif args.optimizer == 'fedyogi':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            tmp_sq = grad ** 2
            tmp_diff = exp_avg_sq - tmp_sq
            exp_avg_sq.add_( - (1 - args.beta2), torch.sign(tmp_diff) * tmp_sq)
            
            denom = exp_avg_sq.sqrt().add_(args.eps)

            step_size = args.lr * lr_decay * math.sqrt(bias_correction2) / bias_correction1
            
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
            
        else:
            exit('unknown optimizer: {}'.format(args.optimizer))


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item() * len(labels)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/total
    return accuracy, loss


def compute_nmse_per_client(global_model,args, net_dataidx_map_train, net_dataidx_map_test,
                                  X_train1, X_test1,
                                  nets=None, device="cpu"):
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.num_users):

        local_model = copy.deepcopy(global_model)
        local_model.eval()


        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        noise_level = 0

        train_dl_local, test_dl_local = csinet_get_divided_dataloader(X_train1, X_test1, args.local_bs,
                                                                              8, dataidxs_train, dataidxs_test,
                                                                              noise_level)




        test_nmse_dB = compute_nmse(local_model, test_dl_local, args, device=device)

        test_results[net_id]['nmse_dB'] = test_nmse_dB


    # test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    tmp = [val['nmse_dB'] for val in test_results.values()]
    test_per_nmse = [round(value, 4) for value in tmp]
    test_all_nmse = sum([10 ** ((val['nmse_dB']) / 10) for val in test_results.values()])
    test_avg_nmse = test_all_nmse / len(test_results)

    return 0, 0, 0, test_per_nmse, test_all_nmse, test_avg_nmse

class channel_truncated(Dataset):
    def __init__(self, matData, dataidxs=None, train=True, transform=None):
        self.matdata = matData
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform

        self.data = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        if self.dataidxs is not None:
            data = self.matdata[self.dataidxs]
        else:
            data = self.matdata

        return data

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.data.shape[0]

def NMSE_cuda(x, x_hat):
    x = x.contiguous().view(len(x), -1)
    x_hat = x_hat.contiguous().view(len(x_hat), -1)
    power = torch.sum(abs(x) ** 2, dim=1)
    mse = torch.sum(abs(x - x_hat) ** 2, dim=1) / power

    return mse


class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, x_hat):
        nmse = NMSE_cuda(x, x_hat)
        # cr_loss = C_R_cuda(x, x_hat)
        # x_np = torch.Tensor.cpu(x).detach().numpy()
        # x_hat_np = torch.Tensor.cpu(x_hat).detach().numpy()
        # score = cal_score(x_np,x_hat_np,128,12)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
            # cr_loss = -torch.mean(cr_loss)
        else:
            nmse = torch.sum(nmse)
            # cr_loss = -torch.sum(cr_loss)
        return nmse  # , cr_loss

class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_noise(signal, snr):
    # signal_power = (1 / int(signal.shape[1])) * k.sum(signal ** 2)
    signal_power = torch.mean(signal ** 2, dim=3, keepdim=True)
    noise_var = signal_power / (10 ** (snr / 10))
    noise = torch.randn(size=signal.shape).cuda()
    noise = torch.sqrt(noise_var).cuda()* noise
    return noise + signal

def csinet_get_divided_dataloader(X_train, X_test, train_bs, test_bs, dataidxs_train, dataidxs_test, noise_level=0, net_id=None, total=0, drop_last=False, apply_noise=False):
    dl_obj = channel_truncated

    train_dataset = dl_obj(matData=X_train, dataidxs=dataidxs_train, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=0, pin_memory=True)
    test_dataset = dl_obj(matData=X_test, dataidxs=dataidxs_test, train=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader

def compute_nmse(model, dataloader, args, device="cpu" ):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]
    criterion_test = NMSELoss(reduction='mean')
    totalLoss = 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, autoencoderInput in enumerate(tmp):
                autoencoderInput = autoencoderInput.to(device)
                out = model(autoencoderInput)
                loss = criterion_test(autoencoderInput, out)
                totalLoss += loss.item()
                averageLoss = totalLoss / len(tmp)
                dB_loss = 10 * math.log10(averageLoss)

    if was_training:
        model.train()

    return dB_loss
