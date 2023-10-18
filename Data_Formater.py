"""
Model Component of ConvLSTM of the Laser Hot Wire AM Anomaly Detection Algorithm
© 2021 Brandon Abranovic, Sulagna Sarkar, Elizabeth Chang-Davidson, and Jack L. Beuth
Carnegie Mellon University
"""


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_video_frame_details(filepath):
    cap = cv2.VideoCapture(filepath)
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return no_of_frames

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def create_repo(path, out_name, object_detection=False):
    #cap = cv2.VideoCapture('N00014-005-011_InLineCam_2.avi')
    #cap = cv2.VideoCapture('N00014-005-021-InLineCam.avi')
    frame_count = get_video_frame_details(path)
    frame_stop = frame_count//2
    cap = cv2.VideoCapture(path)

    #click_x = 494
    #click_y = 393
    click_x = 484
    click_y = 312
    size = 227 #square
    margin1 = (size-1)/2
    margin2 = (size+1)/2

    # ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resize = cv2.resize(gray, (227, 227))
    # resize=resize[np.newaxis,:, :]
    frames_list=[]
    i=1
    j=1
    x_low=445
    x_high=543
    y_low=379
    y_high=475
    # write_dir1 = out_name + '/Test'
    # write_dir = write_dir1 + '/Test1'
    # os.makedirs(write_dir)
    print('Performing video pre-processing steps...')
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if object_detection == True:
        pbar1 = tqdm(total=num_frames_total, desc='Determining average melt pool location across video...', position=0)
        pbar2 = tqdm(total=num_frames_total, desc='Images formatted', position=0)
    if object_detection == False:
        pbar2 = tqdm(total=num_frames_total, desc='Images formatted', position=0)
    if object_detection == True:
        bright_spots_x = 0
        bright_spots_y = 0
        avg_iterator = 0
        checker=[]
        while (cap.isOpened()):
            ii = "{0:03}".format(i)
            ret, frame = cap.read()
            if frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayblur = cv2.GaussianBlur(gray, (95, 95), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayblur)
            if maxVal > 100:
                bright_spots_x += maxLoc[0]
                bright_spots_y += maxLoc[1]
                avg_iterator += 1
                checker.append(True)
            else:
                checker.append(False)
            pbar1.update(1)
        avg_x = bright_spots_x//avg_iterator
        avg_y = bright_spots_y // avg_iterator
        #import ipdb; ipdb.set_trace()
    cap = cv2.VideoCapture(path)
    kk = 0
    while (cap.isOpened()):
        ii = "{0:03}".format(i)
        ret, frame = cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if object_detection == True:
            if checker[kk] == False:
                cropped = gray[int(y_low):int(y_high), int(x_low):int(x_high)]
            if checker[kk] == True:
                cropped = gray[int(avg_y-25):int(avg_y+75), int(avg_x-50):int(avg_x+50)]
        if object_detection == False:
            cropped = gray[int(y_low):int(y_high), int(x_low):int(x_high)]
        frames_list.append(cropped)
        #cv2.imwrite(write_dir+'/'+ii+'.tif', cropped)
        if kk == frame_stop:
            gray=rotate_image(gray, -27)
            print(max(gray.reshape(-1)))
            melt = np.argwhere(gray > 162)
            row, col = melt[-1,:]
            #import ipdb; ipdb.set_trace()
            #col = melt[row, 1]
            print(row, col)
            #col_use = int(np.average(col))
            #import ipdb; ipdb.set_trace()
            print(gray.shape)
            plt.figure(1)
            plt.imshow(gray)
            plt.scatter(col, row)
            plt.show()

            plt.figure(2)
            plt.imshow(gray[row:row+50, col-25:col+25])
            plt.show()
            #import ipdb; ipdb.set_trace()
        i += 1
        kk += 1
        pbar2.update(1)
    # while (cap.isOpened()):
    #     ii = "{0:03}".format(i)
    #     ret, frame = cap.read()
    #     if frame is None:
    #         break
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     if object_detection == True:
    #         grayblur = cv2.GaussianBlur(gray, (95, 95), 0)
    #         (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayblur)
    #         #print(maxVal)
    #         if maxVal < 100:
    #             cropped = gray[int(y_low):int(y_high), int(x_low):int(x_high)]
    #         else:
    #             cropped = gray[int(maxLoc[1]-25):int(maxLoc[1]+75), int(maxLoc[0]-50):int(maxLoc[0]+50)]
    #     if object_detection == False:
    #         cropped = gray[int(y_low):int(y_high), int(x_low):int(x_high)]
    #     frames_list.append(cropped)
    #     cv2.imwrite(write_dir+'/'+ii+'.tif', cropped)
    #     i += 1
    #     pbar2.update(1)
    #print('Pre-processing steps complete!')
    return None #write_dir1, i-1
if __name__ == "__main__":
    create_repo('C:/Users/Brandon/Downloads/QM Control Test 2 FLIR Videos/Block 1/Bead1.avi', 'none')