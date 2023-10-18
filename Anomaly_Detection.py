import torch
import torch.nn as nn
import torch.utils.data as data
from Smarter_Dataset import MeltpoolDataset
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from Data_Formater import create_repo
import shutil
import os
from tqdm import tqdm
import sys
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM


if os.path.isdir('temp_analysis_folder'):
    shutil.rmtree('temp_analysis_folder')

use_cuda = torch.cuda.is_available()
criterion = nn.MSELoss()

n_steps_past = 20
n_steps_ahead = 10

print('ConvLSTM Laser Hot Wire AM Anomaly Detection Algorithm Version 2.0')
print('Provided as a part of the ONR Quality Made Project.')
print('© 2021 Brandon Abranovic, Elizabeth Chang-Davidson, and Jack L. Beuth')
print('Carnegie Mellon University')

if use_cuda:
    print('CUDA enabled GPU device detected, acceleration will be enabled.')
    device = torch.device('cuda:0')
else:
    print('No CUDA enabled GPU detected, CPU will be used. Expect diminished performance.')
    device = torch.device('cpu')
model = EncoderDecoderConvLSTM(nf=64, in_chan=1)

if use_cuda:
    cudnn.benchmark = True
    #model.cuda()
    net = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    crit = torch.nn.DataParallel(criterion, device_ids=list(range(torch.cuda.device_count())))
    #criterion.cuda()

if getattr(sys, 'frozen', False):
     model.load_state_dict(torch.load(os.path.join(sys._MEIPASS,'trained_params_2010w_size.pt'), map_location=device))
else:
     model.load_state_dict(torch.load('trained_params_2010w_size.pt', map_location=device))

model.eval()
video_filename = input("Provide path to video: ")
analyse_path, num_frames = create_repo(video_filename, 'temp_analysis_folder')

print(analyse_path)

test_ds = MeltpoolDataset(analyse_path, num_frames, seq_len = n_steps_past + n_steps_ahead, time_stride=1)
test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, shuffle=False)


frames = []
errors = []
print('Now starting convolutional LSTM analysis of input video...')
recons=[]
inps=[]
pbar = tqdm(total=len(test_dl), desc='Analyzing...', position=0)
crit=nn.MSELoss()

def forward(x):
    x = x.to(device='cuda')
    #output = model(x, future_seq=n_steps_ahead)
    output = net(x, future_seq=n_steps_ahead)
    del x
    return output
for batch_idx, x in enumerate(test_dl):
    #import ipdb;ipdb.set_trace()
    x, y = x[:, 0:n_steps_past, :, :, :], x[:, n_steps_past:, :, :, :]
    x = x.permute(0, 1, 4, 2, 3)
    #y = y.squeeze()

    y_hat = forward(x).squeeze()  # is squeeze neccessary?
    #import ipdb;ipdb.set_trace()
    #mse = torch.norm(x.squeeze().data.view(x.size(0), -1) - y_hat.cpu().data.view(x.size(0), -1), dim=1)
    total_mse=0
    for k in range(10):
        mse = torch.norm(y[:,k,:,:,:].squeeze().data.view(x.size(0), -1) - y_hat[k,:,:].cpu().data.view(x.size(0), -1), dim=1)
        total_mse+=mse
    mse=total_mse/10
    del y_hat
   # import ipdb; ipdb.set_trace()
    errors.extend(mse)
    pbar.update(1)
errors = np.array(errors)

len_plot = int(np.floor(num_frames/(n_steps_past + n_steps_ahead))*20)

errors = errors.reshape(-1, num_frames-29)
s = np.zeros((1,  num_frames-29))
s[0,:] = 1 - (errors[0,:] - np.min(errors[0,:]))/(np.max(errors[0,:]) - np.min(errors[0,:]))

# Test001
plt.figure(1)
plt.xlabel('Video Frame Index')
plt.ylabel('Regularity Score')
plt.title('Bead Analysis Result')

plt.plot(s[0,:])
plt.show()
shutil.rmtree('temp_analysis_folder')


