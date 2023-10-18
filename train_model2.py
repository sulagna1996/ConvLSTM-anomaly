"""
Model Component of ConvLSTM of the Laser Hot Wire AM Anomaly Detection Algorithm
© 2021 Brandon Abranovic, Sulagna Sarkar, Elizabeth Chang-Davidson, and Jack L. Beuth
Carnegie Mellon University
"""
#import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from ucsd_dataset import UCSDAnomalyDataset
from video_CAE import VideoAutoencoderLSTM
import torch.backends.cudnn as cudnn
import numpy as np
# matplotlib notebook
import matplotlib.pyplot as plt
import os
import cv2
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM



#os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = EncoderDecoderConvLSTM(nf=26, in_chan=1)
criterion = nn.MSELoss()

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = True
    model.cuda()
    criterion.cuda()


train_ds = UCSDAnomalyDataset('MP_Train2/Train')#, time_stride=3)
train_dl = data.DataLoader(train_ds, batch_size=12, shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-5)

epochs = 3
#pbar_batch = tqdm(total=len(train_dl),desc='Batch', position=0)
#pbar_epoch = tqdm(total=epochs, desc='Epoch', position=1)
#loss_log = tqdm(total=0, position=2, bar_format='{desc}')
model.train()
print(len(train_dl))
for epoch in range(epochs):
    for batch_idx, x in enumerate(train_dl):
        print(batch_idx)
        #import ipdb; ipdb.set_trace()
        optimizer.zero_grad()
        if use_cuda:
            x = x.cuda()
        y = model(x)
        loss = criterion(y, x)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch {}, iter {}: Loss = {}'.format(
                epoch, batch_idx, loss.item()))
#        loss_log.set_description_str(str(loss.item()))
#        pbar_batch.update(1)
#    pbar_epoch.update(1)
    # torch.save({
    #     'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict()},
    # './snapshot/checkpoint.epoch{}.pth.tar'.format(epoch))
torch.save(model, 'test_Window_Size2_personal.pth')
#model = VideoAutoencoderLSTM()
# model = torch.load('test.pth')
# model.val()
# #
# # test_ds = UCSDAnomalyDataset('MP_Train/Stable_Test')e
# #test_ds = UCSDAnomalyDataset('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')
# test_dl = data.DataLoader(test_ds, batch_size=32, shuffle=False)
# #import ipdb;ipdb.set_trace()
#
# frames = []
# errors = []
# for batch_idx, x in enumerate(test_dl):
#     print(batch_idx)
#     if use_cuda:
#         x = x.cuda()
#     y = model(x)
#     mse = torch.norm(x.cpu().data.view(x.size(0), -1) - y.cpu().data.view(y.size(0), -1), dim=1)
#     errors.append(mse)
# errors = torch.cat(errors).numpy()
#
#
# errors = errors.reshape(-1, 191)
# s = np.zeros((2,191))
# s[0,:] = 1 - (errors[0,:] - np.min(errors[0,:]))/(np.max(errors[0,:]) - np.min(errors[0,:]))
# s[1,:] = 1 - (errors[1,:] - np.min(errors[1,:]))/(np.max(errors[1,:]) - np.min(errors[1,:]))
#
#
#
# # Test001
# plt.plot(s[0,:])
# plt.show()
#
#
# # Test032
# plt.plot(s[1,:])
# plt.show()