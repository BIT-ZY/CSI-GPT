import h5py
import time
import os
import numpy as np
import math
import torch
from scipy.io import savemat
from SWT_final import  SWTCAN,NMSELoss,DatasetFolder
BIT_List = [1024,2048]
SNR_list = [20]
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
iter_loss = []
# for NUM_FEEDBACK_BITS in BIT_List:
for NUM_FEEDBACK_BITS in BIT_List:
    snr = 20
    # NUM_FEEDBACK_BITS = 128
    num = 2
    # m = 16
    # n = 16
    N_BS = 256
    K = 256
    N_ms = 1
    cr = 1 / 8
    mtx_v = N_BS
    mtx_h = int(N_BS * cr)
    input_dim = int(N_BS)
    output_dim = int(N_BS * cr)
    # snr = 20
    CHANNEL_SHAPE_DIM1 = K #K
    CHANNEL_SHAPE_DIM2 = N_BS
    CHANNEL_SHAPE_DIM3 = 2
    # Parameters Setting for Training
    BATCH_SIZE = 64
    EPOCHS = 200
    num_workers = 0  # 2
    LEARNING_RATE = 1e-5
    PRINT_RREQ = 50

    train = 'C:\\Users\\ZY\Desktop\CDLv2\H_train_down_B_6kv2_41.npy'#加载数据集.npy文件
    data_train = np.load(train)
    data_train = data_train.astype('float32')  # 训练变量类型转换
    # iter = 1
    num_data = 6000
    path = './40GHz_CDL-A_model_rho_8/' + str(num_data)
    idx = np.random.choice(np.arange(0, 6000), num_data, replace=False)
    data_train_sample = data_train[idx]
    train_dataset = DatasetFolder(data_train_sample)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)

    # test 环节
    val =  'C:\\Users\\ZY\Desktop\CDLv2\H_test_down_B_2kv2_41.npy'#加载数据集.npy文件
    data_test = np.load(val)
    data_test = data_test.astype('float32')  # 训练变量类型转换

    test_dataset = DatasetFolder(data_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = SWTCAN(num_feedbit=NUM_FEEDBACK_BITS,snr=snr,cr=cr)



    model = model.cuda()



#计算参数量
    params = list(model.named_parameters())
    unique_params = {}
    for name,param in params:
        if param.requires_grad and name not in unique_params:
            unique_params[name] = param.numel()
    num_params = sum(unique_params.values())
    print(num_params)



    criterion = NMSELoss(reduction='mean')
    criterion_test = NMSELoss(reduction='mean')
    class NoamOpt(object):
        "Optim wrapper that implements rate."

        def __init__(self, model_size, factor, warmup, optimizer):#model_size=256, factor=1, warmup=4000
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


    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    opt = NoamOpt(256, 1, 4000, torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))


    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.Adam(model.parameters())
    # =======================================================================================================================
    # =======================================================================================================================
    # Model Training and Saving
    bestLoss = 1
    train_loss_history = []
    test_loss_history = []

    os.makedirs(path,exist_ok=True)
    print('Training for ' + 'snr= '+str(snr) + 'dB SNR ' + str(NUM_FEEDBACK_BITS) + ' bits begin .')
    for epoch in range(EPOCHS):
        t1 = time.time()
        print('========================')
        print('lr:%.4e' % opt.optimizer.param_groups[0]['lr'])
        model.train()
        loss_epoch = 0
        # if epoch == 5:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
        for i, autoencoderInput in enumerate(train_loader):
            autoencoderInput = autoencoderInput.cuda()
            # autoencoderInput_noise = get_noise(autoencoderInput,snr)
            autoencoderOutput = model(autoencoderInput)  #y_noise（batchsize,32,64,2）

            loss = criterion(autoencoderInput,autoencoderOutput)
            opt.optimizer.zero_grad()
            loss.backward()
            opt.step()

            if i % PRINT_RREQ == 0 :
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'  'dB_Loss {dB_loss:.4f}\t'.format(
                    epoch, i, len(train_loader), loss=loss.item(), dB_loss=10*math.log10(loss.item())))
                loss_epoch = loss_epoch + loss.item()
        # Model Evaluating
        average_loss_epoch = loss_epoch / len(train_loader)
        train_loss_history.append(average_loss_epoch)
        model.eval()
        totalLoss = 0

        with torch.no_grad():
            totalLoss = 0
            for i, autoencoderInput in enumerate(test_loader):
                autoencoderInput = autoencoderInput.cuda()
                autoencoderOutput = model(autoencoderInput)
                loss = criterion_test(autoencoderInput, autoencoderOutput)
                totalLoss += loss.item()
            averageLoss = totalLoss / len(test_loader)
            dB_loss = 10*math.log10(averageLoss)
            test_loss_history.append(averageLoss)
            print('Loss %.4f' % averageLoss, 'dB Loss %.4f' % dB_loss)

            if averageLoss < bestLoss:
                # Model saving
                # Encoder Saving

                torch.save({'state_dict': model.encoder.state_dict(
                ), }, path+'/rho_8_SWT_encoder_'+str(NUM_FEEDBACK_BITS)+str(snr)+'v2.pth.tar')
                # Decoder Saving
                torch.save({'state_dict': model.decoder.state_dict(
                ), }, path+'/rho_8_SWT_decoder_'+str(NUM_FEEDBACK_BITS)+str(snr)+'v2.pth.tar')
                print("Model saved")
                bestLoss = averageLoss

        t2 = time.time()

        print('Time: ', t2-t1)
        savemat(path + '/' +str(NUM_FEEDBACK_BITS) + 'num_data='+str(num_data)+'snr='+ str(snr) + '_rho_8_train_loss.mat', {'loss': train_loss_history})
        savemat(path + '/'+ str(NUM_FEEDBACK_BITS) + 'num_data='+str(num_data)+'snr='+str(snr) + '_rho_8_test_loss.mat', {'loss': test_loss_history})

    print('Training for '+str(NUM_FEEDBACK_BITS)+' bits is finished!')
