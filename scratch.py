# from Smarter_Dataset import MeltpoolDataset
# n_steps_past = 20
# n_steps_ahead = 20
# clip_const=39
# analyse_path='FLIR/FLIR_TRAIN/Train/Train8/'
from FLIR_Formatter import Parallel_Preprocessing
import os

# video_filename = 'C:/Users/Brandon/Downloads/QM Control Test 2 FLIR Videos/Block 1/Bead9.avi'
# object_detection = True
# k = Parallel_Preprocessing(video_filename, analyse_path, object_detection)
# # num_frames = k.frame_count - 1
# # test_ds = MeltpoolDataset(analyse_path, num_frames, seq_len = n_steps_past + n_steps_ahead, time_stride=1)

if __name__ == '__main__':
    #path = r'C:\Users\Brandon\Downloads\Jan-17-FLIR-Data'
    path= '/Users/brandonabranovic/Downloads/Jan-17-FLIR-Data/Test' #mac
    vids= os.listdir(path)
    i=1
    for file in vids:
        if file == '.DS_Store':
            continue
        print(file)
        os.mkdir('FLIR_Train_Jan21/Test/Test'+str(i))
        k = Parallel_Preprocessing(path + '/' + file, i, True, True)
        i+=1
        #break