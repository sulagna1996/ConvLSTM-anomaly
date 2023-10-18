"""
Model Component of ConvLSTM of the Laser Hot Wire AM Anomaly Detection Algorithm
© 2021 Brandon Abranovic, Sulagna Sarkar, Elizabeth Chang-Davidson, and Jack L. Beuth
Carnegie Mellon University
"""

import cv2
import time
import subprocess as sp
import multiprocessing as mp
from os import remove
import numpy as np
from tqdm import tqdm

class Parallel_Preprocessing:
    def __init__(self, file_name, mp_detect=True):
        self.mp_detection = mp_detect
        self.file_name = file_name #r'C:\Users\Brandon\Desktop\Video_Demo\N00014-005-033_FrontCam.avi'
        #self.write_dir= 'FLIR/FLIR_TRAIN/Train/Train2/'#'temp_analysis_folder/Test/Test1/'
        self.write_dir = 'temp_analysis_folder/Test/Test1/'
        self.frame_count = self.get_video_frame_details(self.file_name)
        self.num_processes = mp.cpu_count()
        self.frame_jump_unit = self.frame_count // self.num_processes
        print("Video frame count = {}".format(self.frame_count))
        print("Number of CPU: " + str(self.num_processes))
        if self.mp_detection == True:
            #self.mp_pbar= tqdm(total=self.frame_count, desc='Determining average melt pool location across video...',
            #             position=0)
            self.mp_detection_coors_format=self.multi_process_part1()
            self.avg_x = int(np.sum(self.mp_detection_coors_format[:, 0])/np.sum(self.mp_detection_coors_format[:, 2]))
            self.avg_y = int(np.sum(self.mp_detection_coors_format[:, 1])/np.sum(self.mp_detection_coors_format[:, 2]))
            self.checker = self.mp_detection_coors_format[:, 2]
        self.cropped_out=self.multi_process_part2()
        self.multi_process_part3(self.cropped_out)

    def get_video_frame_details(self,filepath):
        cap = cv2.VideoCapture(filepath)
        no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return no_of_frames

    def find_mp(self, frame, fnum):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayblur = cv2.GaussianBlur(gray, (95, 95), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayblur)
        if maxVal > 100:
            bright_spots_x = maxLoc[0]
            bright_spots_y = maxLoc[1]
            checker = True
        else:
            bright_spots_x = 0
            bright_spots_y = 0
            checker = False
        #self.mp_pbar.update(1)
        return bright_spots_x, bright_spots_y, checker, fnum

    def mp_detect_multiprocessing(self, group_number):
        # Read video file
        cap = cv2.VideoCapture(self.file_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jump_unit * group_number)
        proc_frames = 0
        mp_detect_list = []
        try:
            while proc_frames < self.frame_jump_unit:
                ret, frame = cap.read()
                if not ret:
                    break
                fnum = cap.get(cv2.CAP_PROP_POS_FRAMES)
                out = self.find_mp(frame, fnum)
                mp_detect_list.append(out)
                proc_frames += 1
        except:
            # Release resources
            cap.release()
        # Release resources
        cap.release()
        return mp_detect_list

    def multi_process_part1(self):
        #print("Video processing using {} processes...".format(self.num_processes))
        # Paralle the execution of a function across multiple input values
        p = mp.Pool(self.num_processes)
        mp_detection_coors = tqdm(p.imap(self.mp_detect_multiprocessing, range(self.num_processes+1)), total=self.num_processes, position=0, desc='Finding average melt pool position...')
        mp_detection_coors_format = np.asarray([item for sublist in mp_detection_coors for item in sublist])
        #print(type(mp_detection_coors))
        #print(type(mp_detection_coors_format))
        return mp_detection_coors_format

    def colourCrop(self, frame, fnum, x_high, y_high, x_low, y_low):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ii = "{0:03}".format(fnum)
        #print('hello_world2')
        if self.mp_detection == False:
            cropped = gray[int(y_low):int(y_high), int(x_low):int(x_high)]
        #if self.mp_detection == True:
        else:
            # if self.checker[int(fnum)] == False:
            #     #print('false')
            #     cropped = gray[int(y_low):int(y_high), int(x_low):int(x_high)]
            # elif self.checker[int(fnum)] == True:
            #     #print('true')
            #     cropped = gray[int(self.avg_y - 25):int(self.avg_y + 75), int(self.avg_x - 50):int(self.avg_x + 50)]
            cropped = gray[int(self.avg_y - 25):int(self.avg_y + 75), int(self.avg_x - 50):int(self.avg_x + 50)]
        return cropped, ii

    def crop_multiprocessing(self, group_number):
        x_low=445
        x_high=543
        y_low=379
        y_high=475
        cap = cv2.VideoCapture(self.file_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jump_unit * group_number)
        proc_frames = 0
        cropped_list=[]
        try:
            while proc_frames < self.frame_jump_unit:
                ret, frame = cap.read()
                if not ret:
                    break
                #print('hello_world')
                fnum = cap.get(cv2.CAP_PROP_POS_FRAMES)
                out = self.colourCrop(frame, fnum, x_high, y_high, x_low, y_low)
                cropped_list.append(out)
                proc_frames += 1
                #print(cropped_list)
        except:
            # Release resources
            cap.release()
        # Release resources
        cap.release()
        return cropped_list

    def writer(self, params):
        image, name = params
        ii = "{0:03}".format(int(float(name)))
        cv2.imwrite(self.write_dir + ii + '.tif', image)
        #self.write_pbar.update(1)

    def multi_process_part2(self):
        p = mp.Pool(self.num_processes)
        #start_time = time.time()
        #print("Video processing using {} processes...".format(self.num_processes))
        if self.mp_detection == True:
            cropped_out = tqdm(p.imap(self.crop_multiprocessing, range(self.num_processes+1)), total=self.num_processes, position=0, desc='Cropping images...')
        if self.mp_detection == False:
            cropped_out = tqdm(p.imap(self.crop_multiprocessing, range(self.num_processes+1)), total=self.num_processes, position=0, desc='Cropping images...')
            #cropped_out = p.map(self.crop_multiprocessing, range(self.num_processes))
        #import ipdb; ipdb.set_trace()
        o1 = [item for sublist in cropped_out for item in sublist]
        names=[]
        images=[]
        for i in o1:
            names.append(i[1])
            images.append(i[0])
        output = [images, names]
        #import ipdb; ipdb.set_trace()
        return output
    def multi_process_part3(self, cropped_out):
        p = mp.Pool(self.num_processes)
        params = zip(cropped_out[0], cropped_out[1])
        p.map(self.writer, params)




#if __name__ == '__main__':
#    k = Parallel_Preprocessing(r'C:\Users\Brandon\Desktop\Video_Demo\N00014-005-033_FrontCam.avi', False)

# #file_name = r'C:\Users\Brandon\Downloads\N00014_006_003_InLineCam_6.avi'
# file_name = r'C:\Users\Brandon\Desktop\Video_Demo\N00014-005-033_FrontCam.avi'
# frame_count = get_video_frame_details(file_name)
# num_processes = mp.cpu_count()
# frame_jump_unit = frame_count // num_processes
# if __name__ == '__main__':
#     #file_name = r'C:\Users\Brandon\Downloads\N00014_006_003_InLineCam_6.avi'
#     #frame_count = get_video_frame_details(file_name)
#     print("Video frame count = {}".format(frame_count))
#     num_processes = mp.cpu_count()
#     print("Number of CPU: " + str(num_processes))
#     frame_jump_unit =  frame_count// num_processes
#     multi_process(True, None)