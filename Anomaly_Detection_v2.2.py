"""
ConvLSTM Laser Hot Wire AM Anomaly Detection Algorithm Version 2.2
© 2021 Brandon Abranovic, Sulagna Sarkar, Elizabeth Chang-Davidson, and Jack L. Beuth
Carnegie Mellon University
"""

import torch
import torch.nn as nn
import torch.utils.data as data
from Smarter_Dataset import MeltpoolDataset
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm
import sys
from EncoderDecoderConvLSTM import EncoderDecoderConvLSTM
import ntpath
from pathlib import Path
import matplotlib.animation as animation
from PIL import Image
from Data_Formatter2 import Parallel_Preprocessing
import multiprocessing

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
if __name__ == "__main__":
    multiprocessing.freeze_support()
    if os.path.isdir('temp_analysis_folder'):
        shutil.rmtree('temp_analysis_folder')

    use_cuda = torch.cuda.is_available()
    criterion = nn.MSELoss()

    print('ConvLSTM Laser Hot Wire AM Anomaly Detection Algorithm Version 2.2')
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
        model.cuda()
        criterion.cuda()

    print('Select model architecture to use for analysis')
    print('To view a model code guide type "help", to load a custom trained architecture type "custom".')
    while True:
        model_code = input("Code:")
        if model_code not in ("01", "02", "03", "04", "help", "custom"):
            print("Invalid selection, try again.")
        else:
            break

    if model_code=='help':
        while model_code == 'help':
            print('For look back time = 10 frames and look forward time = 10 frames use code 01')
            print('For look back time = 20 frames and look forward time = 20 frames use code 02')
            print('For look back time = 20 frames and look forward time = 10 frames use code 03')
            print('For look back time = 10 frames and look forward time = 20 frames use code 04')
            while True:
                model_code = input("Code:")
                if model_code not in ("01", "02", "03", "04", "help", "custom"):
                    print("Invalid selection, try again.")
                else:
                    break
            if model_code in(["01","02","03","04","custom"]):
                break

    if model_code == "01":
        print('Model 01 selected')
        if getattr(sys, 'frozen', False):
            model.load_state_dict(torch.load(os.path.join(sys._MEIPASS,'trained_params3.pt'), map_location=device))
        else:
            model.load_state_dict(torch.load('trained_params3.pt', map_location=device))
        n_steps_past = 10
        n_steps_ahead = 10
        clip_const = 19
        while True:
            average_perf = input("Use averaged regularity of output performance? (y/n):")
            if average_perf not in ("y", "n"):
                print("Invalid selection, try again.")
            else:
                break

        if average_perf == "y":
            print('Output performance over all 10 predicted frames will be used.')
            averaging = True
            averaging_parameter = 10
        elif average_perf == "n":
            averaging = False
            while True:
                performance_frame = int(input("Select a frame 1-10 to use for prediction performance evaluation:"))
                if performance_frame not in range(1, 11):
                    print("Invalid selection, try again.")
                else:
                    break

    if model_code == "02":
        print('Model 02 selected')
        if getattr(sys, 'frozen', False):
            #model.load_state_dict(torch.load(os.path.join(sys._MEIPASS,'trained_params_20w_size.pt'), map_location=device))
            model.load_state_dict(torch.load(os.path.join(sys._MEIPASS, 'FLIR.pt'), map_location=device))
        else:
            #model.load_state_dict(torch.load('trained_params_20w_size.pt', map_location=device))
            model.load_state_dict(torch.load('FLIR.pt', map_location=device))
        n_steps_past = 20
        n_steps_ahead = 20
        clip_const=39
        while True:
            average_perf = input("Use averaged regularity of output performance? (y/n):")
            if average_perf not in ("y", "n"):
                print("Invalid selection, try again.")
            else:
                break

        if average_perf == "y":
            print('Output performance over all 20 predicted frames will be used.')
            averaging = True
            averaging_parameter = 20
        elif average_perf == "n":
            averaging = False
            while True:
                performance_frame = int(input("Select a frame 1-20 to use for prediction performance evaluation:"))
                if performance_frame not in range(1, 21):
                    print("Invalid selection, try again.")
                else:
                    break

    if model_code == "03":
        print('Model 03 selected')
        if getattr(sys, 'frozen', False):
            model.load_state_dict(torch.load(os.path.join(sys._MEIPASS,'trained_params_2010w_size.pt'), map_location=device))
        else:
            model.load_state_dict(torch.load('trained_params_2010w_size.pt', map_location=device))
        n_steps_past = 20
        n_steps_ahead = 10
        clip_const = 29

        while True:
            average_perf = input("Use averaged regularity of output performance? (y/n):")
            if average_perf not in ("y", "n"):
                print("Invalid selection, try again.")
            else:
                break

        if average_perf == "y":
            print('Output performance over all 10 predicted frames will be used.')
            averaging = True
            averaging_parameter = 10
        elif average_perf == "n":
            averaging = False
            while True:
                performance_frame = int(input("Select a frame 1-10 to use for prediction performance evaluation:"))
                if performance_frame not in range(1, 11):
                    print("Invalid selection, try again.")
                else:
                    break

    if model_code == "04":
        print('Model 04 selected')
        if getattr(sys, 'frozen', False):
            model.load_state_dict(torch.load(os.path.join(sys._MEIPASS,'trained_params_1020w_size.pt'), map_location=device))
        else:
            model.load_state_dict(torch.load('trained_params_1020w_size.pt', map_location=device))
        n_steps_past = 10
        n_steps_ahead = 20
        clip_const = 29
        while True:
            average_perf = input("Use averaged regularity of output performance? (y/n):")
            if average_perf not in ("y", "n"):
                print("Invalid selection, try again.")
            else:
                break
        if average_perf == "y":
            print('Output performance over all 20 predicted frames will be used.')
            averaging = True
            averaging_parameter = 20
        elif average_perf == "n":
            averaging = False
            while True:
                performance_frame = int(input("Select a frame 1-20 to use for prediction performance evaluation:"))
                if performance_frame not in range(1, 21):
                    print("Invalid selection, try again.")
                else:
                    break
    if model_code == "custom":
        print('Custom architecture selected')
        custom_path = input('Provide path to saved model state dict: ')
        # if getattr(sys, 'frozen', False):
        #     model.load_state_dict(
        #         torch.load(os.path.join(sys._MEIPASS, 'trained_params_1020w_size.pt'), map_location=device))
        # else:
        #     model.load_state_dict(torch.load('trained_params_1020w_size.pt', map_location=device))
        in_out_params = os.path.splitext(path_leaf(custom_path))
        n_steps_past = int(in_out_params[0].split('_')[0])
        n_steps_ahead = int(in_out_params[0].split('_')[1])
        clip_const = n_steps_past + n_steps_ahead-1
        while True:
            average_perf = input("Use averaged regularity of output performance? (y/n):")
            if average_perf not in ("y", "n"):
                print("Invalid selection, try again.")
            else:
                break

        if average_perf == "y":
            print('Output performance over all 20 predicted frames will be used.')
            averaging = True
            averaging_parameter = n_steps_ahead
        elif average_perf == "n":
            averaging = False
            while True:
                performance_frame = int(input("Select a frame 1-20 to use for prediction performance evaluation:"))
                if performance_frame not in range(1, n_steps_ahead+1):
                    print("Invalid selection, try again.")
                else:
                    break

    model.eval()

    while True:
        use_object_detection = input("Use melt pool detection? (y/n):")
        if use_object_detection not in ("y", "n"):
            print("Invalid selection, try again.")
        else:
            break

    if use_object_detection == 'y':
        object_detection=True
    if use_object_detection == 'n':
        object_detection = False


    while True:
        print('If you wish to save to folder where video is stored please type: current')
        regularity_log = input("Save regularity analysis to output text file? (y/n/current):")
        if regularity_log not in ("y", "n", "current"):
            print("Invalid selection, try again.")
        else:
            break

    if regularity_log == 'y':
        logging = True

        #while True:
        txt_path = input("Provide filepath to save location for regularity text file:")
        my_file = Path(txt_path)
            #if my_file.is_dir():
            #    break
            #else:
            #    print("Invalid or non-existent directory, try again.")
            #    continue

    if regularity_log == 'n':
        logging = False

    if regularity_log == 'current':
        logging = True

    while True:
        print('If you wish to save to folder where video is stored please type: current')
        animation_output = input("Save animated output plot? (y/n/current):")
        if animation_output not in ("y", "n", "current"):
            print("Invalid selection, try again.")
        else:
            break

    if animation_output == 'y':
        create_animation = True

        #while True:
        anim_path = input("Provide filepath to save location for animation file:")
        my_file = Path(anim_path)
            #if my_file.is_dir():
            #    break
            #else:
            #    print("Invalid or non-existent directory, try again.")
            #    continue

    if animation_output == 'n':
        create_animation = False

    if animation_output == 'current':
        create_animation = True


    fp1 = 'temp_analysis_folder'
    os.makedirs(fp1)
    fp2 = 'temp_analysis_folder/Test'
    os.makedirs(fp2)
    write_dir='temp_analysis_folder/Test/Test1'
    os.makedirs(write_dir)

    video_filename = input("Provide path to video: ")

    plot_title = path_leaf(video_filename)
    #analyse_path, num_frames = create_repo(video_filename, 'temp_analysis_folder', object_detection)
    k = Parallel_Preprocessing(video_filename, object_detection)
    num_frames = k.frame_count - 1
    analyse_path = 'temp_analysis_folder/Test'
    test_ds = MeltpoolDataset(analyse_path, num_frames, seq_len = n_steps_past + n_steps_ahead, time_stride=1)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, shuffle=False)


    frames = []
    errors = []
    print('\nNow starting convolutional LSTM analysis of input video...')
    recons=[]
    inps=[]
    pbar = tqdm(total=len(test_dl), desc='Analyzing...', position=0)
    crit=nn.MSELoss()

    def forward(x):
        x = x.to(device='cuda')
        output = model(x, future_seq=n_steps_ahead)
        del x
        return output
    for batch_idx, x in enumerate(test_dl):
        x, y = x[:, 0:n_steps_past, :, :, :], x[:, n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y_hat = forward(x).squeeze()  # is squeeze neccessary?

        if averaging == False:
            mse = torch.norm(y[:, performance_frame-1, :, :, :].squeeze().data.view(x.size(0), -1) - y_hat[performance_frame-1, :, :].cpu().data.view(x.size(0), -1), dim=1)
        if averaging == True:
            total_mse=0
            for k in range(averaging_parameter):
                mse = torch.norm(y[:, k, :, :, :].squeeze().data.view(x.size(0), -1) - y_hat[k,:,:].cpu().data.view(x.size(0), -1), dim=1)
                total_mse+=mse
            mse=total_mse/averaging_parameter
        del y_hat
        errors.extend(mse)
        pbar.update(1)
    errors = np.array(errors)

    errors = errors.reshape(-1, num_frames-clip_const)
    s = np.zeros((1,  num_frames-clip_const))
    s[0,:] = 1 - (errors[0,:] - np.min(errors[0,:]))/(np.max(errors[0,:]) - np.min(errors[0,:]))


    #LOAD LOG FILE AND CREATE PLOT

    output_filename = os.path.splitext(plot_title)[0]
    data_filename = output_filename.replace("_InLineCam", "") + ".txt"
    data_filename = output_filename.replace("_FrontCam", "") + ".txt"
    print(data_filename)
    data_dir = os.path.dirname(video_filename)
    my_file = Path(data_dir + '/' + data_filename)
    if my_file.is_file():
        frame_nums = np.loadtxt(data_dir + '/' + data_filename, skiprows=2, usecols=5)
    else:
        print("\nVideo metadata txt file not found in expected location. Will default to indexing frames from zero")
        frame_nums=[1]

    frame_plot_range=np.arange(frame_nums[0],frame_nums[0]+len(s[0,:]),1)
    plt.figure(1)
    plt.xlabel('Video Frame Index')
    plt.ylabel('Regularity Score')
    plt.title(plot_title)
    plt.plot(frame_plot_range,s[0,:])
    plt.show()

    ## CREATE LOG FILE IF REQUESTED


    if logging == True:
        if regularity_log == 'current':
            txt_path=data_dir
        output_pathname= txt_path + '/' + output_filename +'_regularity_output.txt'
        f = open(output_pathname, "x")
        f.write("Frame Number" + "   " + "Regularity Score" + "\n")
        j = 0
        for element in s[0,:]:
            f.write(str(frame_nums[0]+j) + "   " + str(element) + "\n")
            j+=1
        f.close()

    #CREATE AND SAVE ANIMATION IF REQUESTED
    if create_animation==True:
        #setup figure
        fig = plt.figure(2, figsize=(16, 6))
        fig.suptitle(plot_title)
        ax1=fig.add_subplot(1,2,1)
        ax2=fig.add_subplot(1,2,2)
        ax2.set(xlabel='Frame Number', ylabel='Regularity Score')

        img = [] # some array of images
        for fr in range(clip_const+1, clip_const+len(s[0, :])):
        #for fr in range(n_steps_past, clip_const + len(s[0, :])):
             im_path = os.path.join('temp_analysis_folder\\Test\\Test1', '{0:03d}.tif'.format(fr))
             img.append(im_path)
        #set up list of images for animation
        ims=[]
        counter = 1
        for time in range(len(s[0,:])-1):
            im_iter = Image.open(img[time])
            im = ax1.imshow(im_iter, cmap='gray')
            im2, = ax2.plot(frame_plot_range[0:time], s[0, 0:time], 'r')
            ims.append([im, im2])

        #run animation
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)
        plt.show()
        if animation_output == 'current':
            anim_path = data_dir
        output_pathname = anim_path + '/' + output_filename + '_animation.gif'
        ani.save(output_pathname)

    shutil.rmtree('temp_analysis_folder')