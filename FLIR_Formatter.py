import os

import cv2
import time
import subprocess as sp
import multiprocessing as mp
from os import remove
import numpy as np
from tqdm import tqdm

class Parallel_Preprocessing:
    def __init__(self, file_name, file_num, evaluate=True, mp_detect=True):
        self.mp_detection = mp_detect
        self.file_name = file_name #r'C:\Users\Brandon\Desktop\Video_Demo\N00014-005-033_FrontCam.avi'
        self.evaluate=evaluate
        if evaluate != True:
            self.write_dir = 'FLIR_Train_Jan21/Train/Test' + str(file_num) + '/'
        else:
            self.write_dir = 'FLIR_Train_Jan21/Test/Test' + str(file_num) + '/'#temp_analysis_folder/Test/Test1/'
        self.frame_count = self.get_video_frame_details(self.file_name)
        self.num_processes = mp.cpu_count()
        self.frame_jump_unit = self.frame_count // (self.num_processes-1)
        print("Video frame count = {}".format(self.frame_count))
        print("Number of CPU: " + str(self.num_processes))
        if self.mp_detection == True:
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

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    def find_mp(self, frame, fnum):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayblur = cv2.GaussianBlur(gray, (95, 95), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayblur)
        gray = self.rotate_image(gray, -27)
        melt = np.argwhere(gray > 162)
        row, col = melt[-1, :]
        if maxVal > 100:
            bright_spots_x = col
            bright_spots_y = row
            checker = True
        else:
            bright_spots_x = 110
            bright_spots_y = 90
            checker = False
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
        # Parallel the execution of a function across multiple input values
        p = mp.Pool(self.num_processes)
        mp_detection_coors = tqdm(p.imap(self.mp_detect_multiprocessing, range(self.num_processes+1)), total=self.num_processes, position=0, desc='Finding average melt pool position...')
        #import ipdb; ipdb.set_trace()
        mp_detection_coors_format = np.asarray([item for sublist in mp_detection_coors for item in sublist])
        #print(type(mp_detection_coors))
        #print(type(mp_detection_coors_format))
        #import ipdb; ipdb.set_trace()
        return mp_detection_coors_format

    def colourCrop(self, frame, fnum, x_high, y_high, x_low, y_low):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.rotate_image(gray, -27)
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
            #cropped = gray[int(self.avg_y - 25):int(self.avg_y + 75), int(self.avg_x - 50):int(self.avg_x + 50)]
            cropped = gray[self.avg_y:self.avg_y + 50, self.avg_x - 25:self.avg_x + 25]
        return cropped, fnum#ii

    def crop_multiprocessing(self, group_number):
        y_low=227-20
        y_high=227
        x_low=180-20
        x_high=180
        cap = cv2.VideoCapture(self.file_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_jump_unit * group_number)
        #print(self.frame_jump_unit * group_number)
        #print(group_number)
        proc_frames = 0
        cropped_list=[]
        try:
            while proc_frames < self.frame_jump_unit:
                #print(proc_frames)
                ret, frame = cap.read()
                if not ret:
                    break
                #if self.evaluate != True:
                    #print(np.average(frame.reshape(-1)))
                    #if max(frame.reshape(-1))<150:
                    #    continue
                    #if np.average(frame.reshape(-1))<55:
                    #    continue
                fnum = cap.get(cv2.CAP_PROP_POS_FRAMES)
                out = self.colourCrop(frame, fnum, x_high, y_high, x_low, y_low)
                cropped_list.append(out)
                proc_frames += 1

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
        p = mp.Pool()#self.num_processes)
        #start_time = time.time()
        #print("Video processing using {} processes...".format(self.num_processes))
        if self.mp_detection == True:
            cropped_out = p.imap(self.crop_multiprocessing, range(self.num_processes))
            #cropped_out = tqdm(p.imap(self.crop_multiprocessing, range(self.num_processes+1)), total=self.num_processes+1, position=0, desc='Cropping images...')
        if self.mp_detection == False:
            cropped_out = tqdm(p.imap(self.crop_multiprocessing, range(self.num_processes)), total=self.num_processes+1, position=0, desc='Cropping images...')
            #cropped_out = p.map(self.crop_multiprocessing, range(self.num_processes))
        #import ipdb; ipdb.set_trace()
        o1 = [item for sublist in cropped_out for item in sublist]
        names=[]

        images=[]
        for i in o1:
            names.append(i[1])
            images.append(i[0])
        output = [images, names]
        return output
    def multi_process_part3(self, cropped_out):
        p = mp.Pool(self.num_processes)
        first = float(cropped_out[1][0])
        #print(first, type(first))
        cropped_out[1][:] = [number - (first-1) for number in cropped_out[1]]
        params = zip(cropped_out[0], cropped_out[1])
        p.map(self.writer, params)





   # k = Parallel_Preprocessing(r'C:\Users\Brandon\Desktop\Video_Demo\N00014-005-033_FrontCam.avi', False)




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