import matplotlib
matplotlib.use('Agg')
import cv2
import sys
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from scipy import stats


def crop_img(img):
    x,y = img.shape;
    x = int(x*2/3)
    y = int(y/2)
    img = img[:x, :y]
    return img

def img_read(path,img_grad,crop, combined_image):
    # try:
    if not img_grad:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (112, 112))
        img = img_to_array(img)
        img = np.squeeze(img,axis=2)
    else:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (112, 112))
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
        if combined_image:
            img = np.concatenate([sobelx, sobely], axis = 1)
        else:
            img = sobelx

    if crop:
        img = crop_img(img)
    return img
    # except Exception as e:
    #     print(path[-10:])
    #     print(e)

def plot_func(columns, sub, save_dir, label, pred, name=None, withGT = True):
    for column in columns:
        if withGT:
            plt.figure(figsize=(20, 12))
            plt.subplot(211)
            plt.plot(np.array(label['time_sec']), np.array(label[column]))
            plt.legend(("ground_truth"))
            plt.title('{} {}'.format(sub, column))
            plt.ylabel(column)
            plt.xlabel('time_sec')
            if columns == ['rotX', 'rotY', 'rotZ']:
                plt.ylim((-0.2, 0.2))
            else:
                plt.ylim((-4, 4))

            plt.subplot(212)
            # plt.figure(figsize=(20, 12))
            plt.plot(np.array(pred['time_sec']), np.array(pred[column]))
            plt.title('{} {}'.format(sub, column))
            plt.legend(("prediction"))
            plt.ylabel(column)
            plt.xlabel('time_sec')
            if columns == ['rotX', 'rotY', 'rotZ']:
                plt.ylim((-0.2, 0.2))
            else:
                plt.ylim((-4, 4))
        else:
            plt.figure(figsize=(20,12))
            plt.plot(np.array(pred['time_sec']), np.array(pred[column]))
            plt.title('{} {}'.format(sub, column))
            plt.legend(("prediction"))
            plt.ylabel(column)
            plt.xlabel('time_sec')
            if columns == ['rotX', 'rotY', 'rotZ']:
                plt.ylim((-0.2, 0.2))
            else:
                plt.ylim((-4, 4))

        if name is None:
            plt.savefig(save_dir + '{}_{}.png'.format(sub, column))
            os.chmod(path=save_dir + '{}_{}.png'.format(sub, column),mode=0o777)
            plt.close('all')
        else:
            plt.savefig(save_dir + '{}_{}_{}.png'.format(sub, column,name))
            os.chmod(path=save_dir + '{}_{}_{}.png'.format(sub, column,name), mode=0o777)
            plt.close('all')


def load_data(store_image_dir,interval_frame,img_grad,crop, combined_image):
    x_input = []
    img_dir = store_image_dir
    frame_0 = img_read(img_dir+"frame_000001.png",img_grad,crop, combined_image)
    print("os.listdir(img_dir):",os.listdir(img_dir)[:5])
    sorted_img_list = sorted(os.listdir(img_dir),key=lambda x: x[-10:-4])
    print("sorted_img_list:",sorted_img_list[:5])
    #predict per 1.3s:
    # for img_path in sorted_img_list[39::39]:

    #predict all frames:
    if interval_frame:
        interval_frame_num = interval_frame
        length = int(39 // interval_frame_num)
        print("intermediate number:",interval_frame_num - 1)
        x_train_list = []

        for i in range(39, len(sorted_img_list)-1,39):
#        for i in range(39,391,39):  # only for test
            if i == 39:
                print("first predict image:",sorted_img_list[i])
            img_num = int(sorted_img_list[i][6:12])

            name_head = img_dir + 'frame_' + str(img_num - length).zfill(6) + '.png'
            f_head = img_read(name_head,img_grad,crop, combined_image) - frame_0

            name_tail = img_dir + 'frame_' + str(img_num).zfill(6) + '.png'
            f_tail = img_read(name_tail,img_grad,crop, combined_image) - frame_0

            f_diff = f_tail - f_head

            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)

            for j in range(interval_frame_num-1):
                try:
                    name_head = img_dir + 'frame_' + str(img_num + length * j).zfill(6) + '.png'
                    # print("head:", name_head)

                    f_head = img_read(name_head,img_grad,crop, combined_image) - frame_0

                    name_tail = img_dir + 'frame_' + str(img_num + length * (j+1)).zfill(6) + '.png'
                    f_tail = img_read(name_tail,img_grad,crop, combined_image) - frame_0
                except:
                    # print("error: ", 'frame_' + str(img_num + length * (j+1)).zfill(6) + '.png')
                    pass

                f_diff = f_tail - f_head

                one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                    np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))

                x_train_list.append(one_input_img)


    x_input = np.array(x_train_list)
    x = np.transpose(x_input,[0,2,3,1])
    print("x.shape", x.shape)
    print("load input done!!")

    return x

def test_single_subject_gpu(data_path = None, model_name = None, subject=None,interval_frame = None,
                            columns_name=None, save_dir_file=None,use_gpu=True,
                            img_grad = None, withGT = None,crop = False, combined_image= False):
    print("test column name:", columns_name)
    print("the name of directory to save img:", save_dir_file)
    if save_dir_file[-1] != "/":
        save_dir_file = save_dir_file + '/'

    # check if directory exist:
    if not os.path.exists(save_dir_file):
        m = os.umask(0)
        os.makedirs(save_dir_file)
        os.umask(m)
    start_image = 0
    if withGT:
 
        print("data_path", data_path)
         
        try:
            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frame")][0] + "/"
        except:
            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frames")][0] + "/"
        
        print("store_image_dir",store_image_dir)
        real_path  = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("head_pose.csv")][0]
        df_real = pd.read_csv(real_path,delimiter=",")
        df_real.to_csv(save_dir_file + "{}_rot_real_short.csv".format(subject))  # eg: ./NC250_01/NC250_01_rot.csv


        print("df_real.shape",df_real.values.shape)
        print("df_real:\n",df_real.head(3))
        filename = data_path + [f for f in os.listdir(data_path) if f.endswith('frame_times.csv')][0]
        with open(filename) as f:
            f.readline()
        #    print(filename)
            for line in f:
                if float(line.split(",")[-1]) < 0:
                    start_image += 1
                else:
                    break
    else:
        try:
            #store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frame")][0] + "/"
            store_image_dir = data_path +  [f for f in os.listdir(data_path) if f.endswith("frame")][0] + "/"
        except:
#            store_image_dir = data_path + "/" + [f for f in os.listdir(data_path) if f.endswith("frames")][0] + "/"
            store_image_dir = data_path  + [f for f in os.listdir(data_path) if f.endswith("frames")][0] + "/"
        print("store_image_dir:",store_image_dir)

    

    print('start_image: {}'.format(start_image))
    # load data
    x_input = load_data(store_image_dir,img_grad = img_grad, interval_frame = interval_frame,crop = crop, combined_image= combined_image)
    x_input = x_input[start_image:]
    print("x_input.shape",x_input.shape)

    #perdict all:
    time_all = np.arange(start=0,stop=len(x_input)) * 0.033333*(39//interval_frame) +1.3
    time_all = list(map(lambda x: round(x,2),time_all))
    time_all = np.array(time_all)
    time_all = time_all.reshape(-1,1)
    # print("time_all:{}\nlength:{}\n".format(time_all[:5],len(time_all)))

    # load model
    if use_gpu:
        # use GPU
        gpu_no = 0
        with tf.device("/gpu:" + str(gpu_no)):
            print("this is to run gpu")
            model = model_name
            y_pred = model.predict(x_input)
            # K.clear_session()
    else:
        model = model_name
        y_pred = model.predict(x_input)
        # K.clear_session()

    print("y_pred.shape:",y_pred.shape)
    pred_train = pd.DataFrame(np.concatenate((y_pred, time_all), axis=1),
                              columns = columns_name + ['time_sec'])
    # to CSV file: note exlude time column
    if columns_name == ["rotX", "rotY", "rotZ"]:
        pred_train.to_csv(save_dir_file + "{}_rot_pred_long.csv".format(subject)) #eg: ./NC250_01/NC250_01_rot.csv
    else:
        pred_train.to_csv(save_dir_file + "{}_trans_pred_long.csv".format(subject))

    if columns_name == ["rotX", "rotY", "rotZ"]:
        pred_train_short = pred_train.iloc[::39,:]
        pred_train_short.to_csv(save_dir_file + "{}_rot_pred_short.csv".format(subject))
        new_real = {}
        for column in ["rotX", "rotY", "rotZ","time_sec"]:
            new_real[column] = np.interp(pred_train['time_sec'].values, df_real['time_sec'].values, df_real[column].values)
        if withGT:
            df_new_real = pd.DataFrame(new_real)
            df_new_real.to_csv(save_dir_file + "/{}_rot_real_long.csv".format(sub))
    else:
        pred_train_short = pred_train.iloc[::39,:]
        pred_train_short.to_csv(save_dir_file + "{}_trans_pred_short.csv".format(subject))
        new_real = {}
        for column in ["transX", "transY", "transZ","time_sec"]:
            new_real[column] = np.interp(pred_train['time_sec'].values, df_real['time_sec'].values, df_real[column].values)
        if withGT:
            df_new_real = pd.DataFrame(new_real)
            df_new_real.to_csv(save_dir_file + "/{}_trans_real_long.csv".format(sub))
        
    if withGT:
        # print("enter with Ground truth")
        plot_func(columns = columns_name, sub = subject, save_dir = save_dir_file, label = df_real, pred = pred_train,
                  withGT = withGT)
    # else:
    #     print("enter No Ground truth")
    #     plot_func(columns=columns_name, sub=subject, save_dir=save_dir_file, label=None, pred = pred_train,
    #               withGT = withGT)

    if withGT:
        return pred_train, df_real, df_new_real
    else:
        return pred_train


def weighted_MAE_rot(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*10,tf.ones_like(y_true,dtype="float32"))
    # y_weight = y_true*10000
    return K.mean(K.abs((y_pred - y_true)*y_weight))  #modify： if <10^-4, *1; else 100  (233,234,244,245)

def weighted_MAE_trans(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*100*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    # y_weight = y_true*10000
    return K.mean(K.abs((y_pred - y_true)*y_weight))

def weighted_MSE_trans(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
#     y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*10*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    # y_weight = y_true*10000
    return K.mean(K.square(y_pred -y_true)*y_weight, axis=-1)

def weighted_MSE_rot(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)

    # ### New weight!!!code report error
    # y_weight = tf.where(condition,
    #                     tf.greater(tf.ones_like(y_true,dtype="float32")*10,tf.ones_like(y_true,dtype="float32")),
    #                     tf.ones_like(y_true,dtype="float32"))

    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*200*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    return K.mean(K.square(y_pred -y_true)*y_weight, axis=-1)

################# main #################################################################################################

def main_test_NoGT(interval_frame, img_grad, withGT, data_path, save_path,
                   judge_rot_trans,model,crop,sub, combined_image):
    currentDT = datetime.datetime.now()
    YMD = "{}{}{}".format(currentDT.year, currentDT.month, currentDT.day)
    HMS = "{}{}{}".format(currentDT.hour, currentDT.minute, currentDT.second)
    start = time.time()
    print('data_path: ',data_path)
    # choose to use gpu or cpu
    use_gpu = True
    if not withGT:        
        VIDEO_FILE = data_path+[f for f in os.listdir(data_path) if '.mp4' in f][0]   # /u/erdos/csga/jzhou40/MRI_Project/sub-NC999_head_pose_data_zhao/sub-NC999_ses-20190702_task-TASK_acq-normal_run-01_bold_video_cropped.mp4'
        print('VIDEO_FILE: ',VIDEO_FILE)

        folder_name = VIDEO_FILE.split("/")[-1].rstrip()[:-4]+"_frames"  # sub-NC999_ses-20190702_task-TASK_acq-normal_run-01_bold_video_cropped_frames
        print('folder_name: ',folder_name)
        folder_path = data_path + folder_name+'/'
        # print('folder_path:',folder_path)
        # if os.path.exists(folder_path) and len(os.listdir(folder_path)) < 10:
        #     print("existing directory for image cropped but it's null!")
        #     os.system("rm -r " + folder_path)
        #     print("remove folder")
        # elif os.path.exists(folder_path):
        if not os.path.exists(folder_path):
            print("Cropping the video to images!")
            m = os.umask(0)
            print("create new folder to store img")
            os.makedirs(folder_path)
            os.umask(m)
            print('create folder')
            print('********enter *****')
            ffmpeg_command = "ffmpeg -i " + VIDEO_FILE+' '+ folder_path +"frame_%06d.png"
            print("ffmpeg_command:",ffmpeg_command)
            os.system(ffmpeg_command)

    save_dir_file = save_path
    print("\n save dir =======", save_dir_file)
    if not os.path.exists(save_dir_file):
        m = os.umask(0)
        os.makedirs(save_dir_file)
        os.umask(m)

    if judge_rot_trans == "rot":
        columns_name = ["rotX", "rotY", "rotZ"]
    else:
        columns_name = ["transX", "transY", "transZ"]

    # use gpu
    print("Using GPU !")
    print("withGT:",withGT)

    if withGT:
        pred_rot, df_real, df_new_real = test_single_subject_gpu(data_path=data_path, model_name=model,
                                                       subject=sub,columns_name=columns_name,
                                                       save_dir_file=save_dir_file, use_gpu=use_gpu,
                                                       img_grad=img_grad, interval_frame=interval_frame,
                                                       withGT=withGT, crop=crop, combined_image = combined_image)

        # calc pvalue and r squared
        r_list = []
        p_list = []
        r_list_long = []
        p_list_long = []

        for column in columns_name:
            pred_val_short = pred_rot[column].values[::39]
            real_val_short = df_real[column].values[1:]

            pred_val_long = pred_rot[column].values
            real_val_long = df_new_real[column].values[39:]

            min_ = min(len(pred_val_short), len(real_val_short))
            min_long = min(len(pred_val_long), len(real_val_long))

            slope, intercept, r_value, p_value, std_err = stats.linregress(real_val_short[:min_].astype(float), pred_val_short[:min_].astype(float))
            slope, intercept, r_value_long, p_value_long, std_err = stats.linregress(real_val_long[:min_long].astype(float),
                                                                                     pred_val_long[:min_long].astype(float))

            r_list.append(r_value ** 2)
            p_list.append(p_value)
            r_list_long.append(r_value_long ** 2)
            p_list_long.append(p_value_long)

        return r_list,p_list, r_list_long, p_list_long

    else:
        print('withoutGT')
        print(columns_name)
        pred_rot = test_single_subject_gpu(data_path=data_path, model_name=model,subject=sub,
                                           columns_name=columns_name, save_dir_file=save_dir_file, use_gpu=use_gpu,
                                           img_grad=img_grad, interval_frame=interval_frame, withGT=withGT, crop=crop,
                                           combined_image = combined_image)

        print("completed rotation part")
        


    # # use gpu
    # pred_trans = test_single_subject_gpu(store_image_dir = "./{}".format(folder_name), model_name = model_trans,
    #                                          subject="{}".format(VIDEO_FILE.split("/")[-1].rstrip()[:-4]),
    #                                          columns_name=["transX", "transY", "transZ"], save_dir_file=save_dir_file,use_gpu=use_gpu)
    # print("completed translation part")

    # combined rotation and translation
    # result = pd.concat([pred_rot.iloc[:,:-1],pred_trans],axis = 1)
    # result.to_csv("./{}/rot_trans.csv".format(YMD,HMS))
    end = time.time()
    print("time_cost:", end - start)



if __name__ == "__main__":
    ###############################################  parameters ###########################

    interval_frame = 39  # true or false
    print("intermeidate:",interval_frame)

    img_grad = True
    print("image loading:", img_grad)

    judge_rot_trans = "rot"
    print("rot or trans:",judge_rot_trans)

    crop = False
    combined_image = True

    data_path = sys.argv[1] # load data
    print("data_path:",data_path)

    save_path = sys.argv[2]  # save results

    if not os.path.exists(save_path):
        m = os.umask(0)
        os.makedirs(save_path)
        os.umask(m)

    if save_path[-1] != "/":
        save_path = save_path + "/"
    print("save_path:", save_path)

    try:
        model_location = sys.argv[3]
    except:
        pass
    print("model location:", sys.argv[3])


    # test_list_withGT = [248, 249, 250, 251, 252, 254, 255, 150, 232, 233, 234, 235, 236, 237, 239, 240, 242, 243, 244, 245, 246, 247]#, 243, 244, 245, 246]
    test_list_withGT = [248, 249] 
    test_list_noGT = [999,888]

    df_r_p_short = {}
    df_r_p_long = {}
    ################################################################################################################
    # load model:
    
    if judge_rot_trans == "rot":
        print("load model for columns: rotX , rotY, rotZ")
        model = load_model(model_location, custom_objects={'weighted_MSE_rot': weighted_MSE_rot})
    else:
        print("load model for columns: transX, transY, transZ")
        model = load_model(model_location, custom_objects={'weighted_MSE_trans': weighted_MSE_trans})
    print("have loaded model!")

    if data_path[-3:] != "mp4":
        if data_path[-1] != "/":
            data_path = data_path + "/"
        # for sub in test_list_withGT:
        for i in range(len(test_list_withGT)):
            # try:
            sub = test_list_withGT[i]
            print("=======with GT: this is sub:", sub, flush=True)
            sub_path = data_path + [f for f in os.listdir(data_path) if str(sub) in f][0] + "/"
            save_sub_path = save_path + "{}_{}_{}/".format("noskip","gradient", "full")
            if save_sub_path[-1] != "/":
                save_sub_path = save_sub_path + "/"
            r_list,p_list, r_list_long, p_list_long = main_test_NoGT(interval_frame, img_grad, True, sub_path, save_sub_path, judge_rot_trans,
                                        model, crop,str(sub),combined_image)
            df_r_p_short[sub] = r_list + p_list
            df_r_p_long[sub] = r_list_long + p_list_long

            tmp = pd.DataFrame(df_r_p_short).transpose()
            if judge_rot_trans == "rot":
                tmp.columns = ["rotx_r_2", "rotY_r_2", "rotZ_r_2", "rotX_P", "rotY_P", "rotZ_P"]
            else:
                tmp.columns = ["transX_r_2", "transY_r_2", "transZ_r_2", "transX_P", "transY_P", "transZ_P"]
            tmp = tmp.sort_index()
            print("short_r2:\n",tmp, flush=True)
            print("saving to r-p file", flush=True)
            tmp.to_csv(save_path + "short_r2_p_value.csv")

            tmp_long = pd.DataFrame(df_r_p_long).transpose()
            if judge_rot_trans == "rot":
                tmp_long.columns = ["rotx_r_2", "rotY_r_2", "rotZ_r_2", "rotX_P", "rotY_P", "rotZ_P"]
            else:
                tmp_long.columns = ["transX_r_2", "transY_r_2", "transZ_r_2", "transX_P", "transY_P", "transZ_P"]
            tmp_long = tmp_long.sort_index()
            print("\nlong_r2:\n", tmp_long,flush=True)
            print("saving to r-p file", flush=True)
            tmp_long.to_csv(save_path + "long_r2_p_value.csv")

            # except:
            #     print("something wrong with this sub:",sub, flush=True)
            #     pass

        for i in range(len(test_list_noGT)):
            try:
                sub = test_list_noGT[i]
                print("========without GT: this is sub:", sub)
                sub_path = data_path + [f for f in os.listdir(data_path) if str(sub) in f][0] + "/"  #'/u/erdos/csga/jzhou40/MRI_Project/sub-NC999_head_pose_data_zhao/'
                
                print("========subject path", sub_path)
                save_sub_path = save_path + "{}_{}_{}/".format("noskip","gradient", "full")
                if save_sub_path[-1] != "/":
                    save_sub_path = save_sub_path + "/"
                pred_rot = main_test_NoGT(interval_frame, img_grad, False, sub_path, save_sub_path, judge_rot_trans,
                                            model,crop, str(sub),combined_image)
                
            except:
                pass
    else:

        sub = test_list_noGT[0]
        save_path = save_path + "{}_{}_{}_{}/".format(sys.argv[1], sys.argv[2],crop_0,save_path.split("/")[-1])
        main_test_NoGT(interval_frame, img_grad, True, data_path, save_path, judge_rot_trans, model,
                       crop, str(sub),combined_image)
    print("completed")

