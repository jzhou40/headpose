#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:53:53 2019

@author: neil:
"""
import pandas as pd
import numpy as np
import cv2
import os
from keras.preprocessing.image import img_to_array
import time

def data_load_single_subject(columns_name = None, test_subject=150, test_sample=100, interval_frame=False,Whole_data =False
                             ,path = None,img_grad = True, crop= False, combined_image = False):
    """

    :param test_subject: test_subjects_name
    :param test_sample:  the size of test_samples
    :param group_size:  if group_size==2: (fn-f0,fn-1 - f0, fn-fn-1)
    :param interval_frame: True/False; if True, with interval frames based on linear model
    :param Whole_data: True: do not cut dataset into train and test (use whole data as test or train)
    :return: x_train, y_train, x_test, y_test
    """
    # if path is None:
    #path = "/u/erdos/students/hyuan11/MRI/gpfs/data/epilepsy/mri/hpardoe/head_pose_datasets/sub-NC{}_head_pose_data_zhao/".format(test_subject)
    path = "/u/erdos/csga/jzhou40/MRI_Project/sub-NC{}_head_pose_data_zhao/".format(test_subject)
    # else:
    #     path = None
    print("dp:the path to load data:",path)
    start_time = time.time()
    df_label, df_img_name, img_dir = load_data(path,columns_name)
    df_merged = pd.merge(df_label, df_img_name, how='left', on='time_sec')

    if not Whole_data:
        print("\n5 min training and 2 mins testing per subject.")
        x_train, y_train, x_test, y_test = get_feature(df_merged, img_dir,test_sample=test_sample,
                                                       interval_frame = interval_frame,img_grad = img_grad,
                                                       crop = crop, combined_image = combined_image)
        test_time_arr = y_test[:, -1].reshape(-1,1)
        train_time_arr = y_train[:, -1].reshape(-1,1)
        y_test = y_test[:, :-1]
        y_train = y_train[:, :-1]
        end_time = time.time()
        print("dp:load image time cost:", end_time - start_time)
        return x_train, y_train, x_test, y_test, test_time_arr, train_time_arr
    else:
        x, y = get_feature_all_data(df_merged = df_merged, img_dir = img_dir, interval_frame=interval_frame,
                                    img_grad = img_grad, crop = crop, combined_image = combined_image)
        time_arr = np.arange(len(y))*1.3 + 3.9 #y[:, -1].reshape(-1, 1)
        y = y[:, :-1]
        end_time = time.time()
        print("dp:load image time cost:", end_time - start_time)
        return x, y, time_arr

def load_data(path,columns_name):
    """

    :param path: dataset path
    :return: dataFrame: data(rot & trans values, time_sec);
             dataFrame: pic_name(img_names & time_sec);
             img_dir: path to image directory
    """
    # get label, image_name and image_directory
    label_name = path + [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
    feature_name = path + [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
    img_dir = path + [f for f in os.listdir(path) if f.endswith('cropped_frames')][0] + '/'

    # load csv file
    df_label = pd.read_csv(label_name, delimiter=',')
    df_img_name = pd.read_csv(feature_name, delimiter=',')

    # match time between frame_times.csv and head_pose.csv
    df_img_name['time_sec'] = round(df_img_name['time_sec'], 2)
    df_img_name['file'] = list(map(lambda x: x[-16:], df_img_name['file']))

    # drop rows when the time in frame_time.csv cannot find in head_pose
    df_label = df_label[df_label["time_sec"] >= df_img_name.loc[0, "time_sec"]].reset_index(drop=True)
    #only focus on rotation part
    # df_label = df_label.loc[:, ["rotX","rotY","rotZ","time_sec"]]
    print("load_data func: columns_name",columns_name)
    columns = columns_name + ["time_sec"]
    df_label = df_label.loc[:,columns]

    return df_label, df_img_name, img_dir

def get_feature(df_merged, img_dir,test_sample, interval_frame,img_grad, crop, combined_image):
    """

    :param df_merged: img_name, rotation parts and time_sec
    :param img_dir: path to image directory
    :param test_sample: the number of test dataset
    :param interval_frame: Boolean or int, if True, with interval frames
    :return: x_train, y_train, x_test, y_test
    """
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    frame_0 = img_read_img_gradient(img_dir + df_merged.loc[0, "file"],img_grad, crop, combined_image)
    print("dp:image_initial:{}".format(df_merged.loc[0, "file"]))
    print("dp:image_shape:", frame_0.shape)
    #######################################################################unmatched
    # load last part of data as testing dataset(size = test_sample)
    # #(fn-f0,fn-2 - f0,fn-4 -f0,fn-6 -f0,fn-8 -f0, fn-10 -f0）
    # for test_i in reversed(range(-1, -test_sample - 1, -1)) :  # -100,-99,...,-1
    #
    #     if test_i == -test_sample :
    #         print("dp:test_img_head:{}".format(df_merged.loc[len(df_merged) + test_i - 1, "file"]))
    #         print("dp:test_img_tail:{}\n".format(df_merged.loc[len(df_merged) + test_i, "file"]))
    #     if interval_frame :
    #         interval_frame_num = interval_frame
    #         length = int((int(df_merged.loc[1, "file"][-10 :-4]) - int(
    #             df_merged.loc[0, "file"][-10 :-4])) / interval_frame_num)  # 39--1
    #     else :
    #         length = 1
    #     test_img_num = int(df_merged.loc[len(df_merged) + test_i, "file"][-10 :-4])
    #     test_i_name = img_dir + 'frame_' + str(test_img_num).zfill(6) + '.png'
    #     test_i_1_name = img_dir + 'frame_' + str(test_img_num - length*2).zfill(6) + '.png'
    #     test_i_2_name = img_dir + 'frame_' + str(test_img_num - length * 4).zfill(6) + '.png'
    #     test_i_3_name = img_dir + 'frame_' + str(test_img_num - length * 8).zfill(6) + '.png'
    #     test_i_4_name = img_dir + 'frame_' + str(test_img_num - length * 10).zfill(6) + '.png'
    #     test_i_5_name = img_dir + 'frame_' + str(test_img_num - length * 12).zfill(6) + '.png'
    #     f_head1 = img_read_img_gradient(test_i_5_name, img_grad, crop, combined_image) - frame_0
    #     f_head2 = img_read_img_gradient(test_i_4_name, img_grad, crop, combined_image) - frame_0
    #     f_head3 = img_read_img_gradient(test_i_3_name, img_grad, crop, combined_image) - frame_0
    #     f_head4 = img_read_img_gradient(test_i_2_name, img_grad, crop, combined_image) - frame_0
    #     f_head5 = img_read_img_gradient(test_i_1_name, img_grad, crop, combined_image) - frame_0
    #     f_tail = img_read_img_gradient(test_i_name, img_grad, crop, combined_image) - frame_0
    #
    #     one_input_img = ([f_head1, f_head2, f_head3, f_head4, f_head5, f_tail] - np.min(
    #         [f_head1, f_head2, f_head3, f_head4, f_head5, f_tail])) / (
    #                             np.max([f_head1, f_head2, f_head3, f_head4, f_head5, f_tail]) - np.min(
    #                         [f_head1, f_head2, f_head3, f_head4, f_head5, f_tail]))
    #     x_test_list.append(one_input_img)
    #
    #     y_test_list.append(df_merged.iloc[test_i, :-1])
    ########################################################
#######################################################################
    # load last part of data as testing dataset(size = test_sample)
    # #(fn-f0,fn-2 - f0,fn-4 -f0,fn-6 -f0,fn-8 -f0, fn-10 -f0）
    # for test_i in reversed(range(-1,-test_sample-1,-1)): #-100,-99,...,-1
    #
    #     if test_i == -test_sample:
    #         print("dp:test_img_head:{}".format(df_merged.loc[len(df_merged) + test_i -1, "file"]))
    #         print("dp:test_img_tail:{}\n".format(df_merged.loc[len(df_merged) + test_i, "file"]))
    #     if interval_frame:
    #         interval_frame_num = interval_frame
    #         length = int((int(df_merged.loc[1, "file"][-10 :-4]) - int(df_merged.loc[0, "file"][-10 :-4])) / interval_frame_num)  # 39--1
    #     else:
    #         length = 1
    #     test_img_num = int(df_merged.loc[len(df_merged) + test_i, "file"][-10 :-4])
    #     test_i_name = img_dir + 'frame_' + str(test_img_num).zfill(6) + '.png'
    #     test_i_1_name = img_dir + 'frame_' + str(test_img_num - length).zfill(6) + '.png'
    #     test_i_2_name = img_dir + 'frame_' + str(test_img_num - length*2).zfill(6) + '.png'
    #     test_i_3_name = img_dir + 'frame_' + str(test_img_num - length*3).zfill(6) + '.png'
    #     test_i_4_name = img_dir + 'frame_' + str(test_img_num - length*4).zfill(6) + '.png'
    #     test_i_5_name = img_dir + 'frame_' + str(test_img_num - length*5).zfill(6) + '.png'
    #     f_head1 = img_read_img_gradient(test_i_5_name, img_grad, crop, combined_image) - frame_0
    #     f_head2 = img_read_img_gradient(test_i_4_name, img_grad, crop, combined_image) - frame_0
    #     f_head3 = img_read_img_gradient(test_i_3_name, img_grad, crop, combined_image) - frame_0
    #     f_head4 = img_read_img_gradient(test_i_2_name, img_grad, crop, combined_image) - frame_0
    #     f_head5 = img_read_img_gradient(test_i_1_name, img_grad, crop, combined_image) - frame_0
    #     f_tail = img_read_img_gradient(test_i_name, img_grad, crop, combined_image) - frame_0
    #
    #     one_input_img = ([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail] - np.min([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail])) / (
    #                         np.max([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail]) - np.min([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail]))
    #     x_test_list.append(one_input_img)
    #
    #     y_test_list.append(df_merged.iloc[test_i, :-1])
########################################################
    #each instance includes three frames: (fn-f0, fn-1 -f0, fn - fn-1)
    for test_i in reversed(range(-1, -test_sample - 1, -1)) :

        if test_i == -test_sample :
            print("dp:test_img_head:{}".format(df_merged.loc[len(df_merged) + test_i - 1, "file"]))
            print("dp:test_img_tail:{}\n".format(df_merged.loc[len(df_merged) + test_i, "file"]))

        # test_img_num = int(df_merged.loc[len(df_merged) + test_i, "file"][-10 :-4])
        # test_i_name = img_dir + 'frame_' + str(test_img_num).zfill(6) + '.png'
        # test_i_1_name = img_dir + 'frame_' + str(test_img_num - 1).zfill(6) + '.png'

        # f_head = img_read_img_gradient(test_i_1_name, img_grad, crop, combined_image) - frame_0
        # f_tail = img_read_img_gradient(test_i_name, img_grad, crop, combined_image) - frame_0
        f_head = img_read_img_gradient(img_dir + df_merged.loc[len(df_merged) + test_i -1, "file"],img_grad, crop, combined_image) - frame_0
        f_tail = img_read_img_gradient(img_dir + df_merged.loc[len(df_merged) + test_i, "file"],img_grad, crop, combined_image) - frame_0
        
        f_diff = f_tail - f_head

        one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
        x_test_list.append(one_input_img)

        y_test_list.append(df_merged.iloc[len(df_merged) + test_i, :-1])

        ###################################
    x_test = np.array(x_test_list)
    print("dp:x_test.shape:",x_test.shape)
    if img_grad:
        x_test = np.transpose(x_test,[0,2,3,1])
        print("dp:x_test.shape", x_test.shape)
    else:
        # x_test = np.squeeze(x_test, axis=4)
        x_test = np.transpose(x_test, [0, 2, 3, 1])
        print("dp:x_test.shape", x_test.shape)

    y_test = np.array(y_test_list)
    # previous 5 img as a group
##################################################
    # if interval_frame:
    #     interval_frame_num = interval_frame
    #     length = int((int(df_merged.loc[1, "file"][-10:-4]) - int(df_merged.loc[0, "file"][-10:-4])) / interval_frame_num)  #39--1
    #     print("dp:length:",length)

        # for train_i in range(1, len(df_merged) - test_sample-1):
        #     if train_i == 1:
        #         print("first predict image:",df_merged.loc[train_i, "file"])
        #     img_num = int(df_merged.loc[train_i, "file"][-10:-4])
        #     label_head = df_merged.iloc[train_i, :-1]
        #
        #     label_diff = (df_merged.iloc[train_i+1, :-1] - df_merged.iloc[train_i, :-1])/interval_frame_num
        #
        #     name_head1 = img_dir + 'frame_' + str(img_num - length*5).zfill(6) + '.png'
        #     name_head2 = img_dir + 'frame_' + str(img_num - length*4).zfill(6) + '.png'
        #     name_head3 = img_dir + 'frame_' + str(img_num - length*3).zfill(6) + '.png'
        #     name_head4 = img_dir + 'frame_' + str(img_num - length*2).zfill(6) + '.png'
        #     name_head5 = img_dir + 'frame_' + str(img_num - length*1).zfill(6) + '.png'
        #     name_tail = img_dir + 'frame_' + str(img_num).zfill(6) + '.png'
        #     f_head1 = img_read_img_gradient(name_head1, img_grad, crop, combined_image) - frame_0
        #     f_head2 = img_read_img_gradient(name_head2, img_grad, crop, combined_image) - frame_0
        #     f_head3 = img_read_img_gradient(name_head3, img_grad, crop, combined_image) - frame_0
        #     f_head4 = img_read_img_gradient(name_head4, img_grad, crop, combined_image) - frame_0
        #     f_head5 = img_read_img_gradient(name_head5, img_grad, crop, combined_image) - frame_0
        #     f_tail = img_read_img_gradient(name_tail,img_grad, crop, combined_image) - frame_0
        #
        #     one_input_img = ([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail] - np.min([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail])) / (
        #                         np.max([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail]) - np.min([f_head1,f_head2,f_head3,f_head4,f_head5,f_tail]))
        #     x_train_list.append(one_input_img)
        #
        #     y_train_list.append(label_head)
        #     for j in range(interval_frame_num-1):
        #         name_head1 = img_dir + 'frame_' + str(img_num + length * (j - 4)).zfill(6) + '.png'
        #         name_head2 = img_dir + 'frame_' + str(img_num + length * (j - 3)).zfill(6) + '.png'
        #         name_head3 = img_dir + 'frame_' + str(img_num + length * (j - 2)).zfill(6) + '.png'
        #         name_head4 = img_dir + 'frame_' + str(img_num + length * (j - 1)).zfill(6) + '.png'
        #         name_head5 = img_dir + 'frame_' + str(img_num + length * j).zfill(6) + '.png'
        #         name_tail = img_dir + 'frame_' + str(img_num + length * (j + 1)).zfill(6) + '.png'
        #
        #         f_head1 = img_read_img_gradient(name_head1, img_grad, crop, combined_image) - frame_0
        #         f_head2 = img_read_img_gradient(name_head2, img_grad, crop, combined_image) - frame_0
        #         f_head3 = img_read_img_gradient(name_head3, img_grad, crop, combined_image) - frame_0
        #         f_head4 = img_read_img_gradient(name_head4, img_grad, crop, combined_image) - frame_0
        #         f_head5 = img_read_img_gradient(name_head5, img_grad, crop, combined_image) - frame_0
        #         f_tail = img_read_img_gradient(name_tail,img_grad, crop, combined_image) - frame_0
        #
        #         one_input_img = ([f_head1, f_head2, f_head3, f_head4, f_head5, f_tail] - np.min([f_head1, f_head2, f_head3, f_head4, f_head5, f_tail])) / (
        #                         np.max([f_head1, f_head2, f_head3, f_head4, f_head5, f_tail]) - np.min([f_head1, f_head2, f_head3, f_head4, f_head5, f_tail]))
        #         x_train_list.append(one_input_img)
        #         y_train_list.append(label_head + label_diff*(j+1))
########################################################
    if interval_frame:
        interval_frame_num = interval_frame
        length = int((int(df_merged.loc[1, "file"][-10:-4]) - int(df_merged.loc[0, "file"][-10:-4])) / interval_frame_num)  #39--1
        print("dp:length:",length)

        for train_i in range(1, len(df_merged) - test_sample-1):
            if train_i == 1:
                print("first predict image:",df_merged.loc[train_i, "file"])
            img_num = int(df_merged.loc[train_i, "file"][-10:-4])
            label_head = df_merged.iloc[train_i, :-1]
            label_diff = (df_merged.iloc[train_i+1, :-1] - df_merged.iloc[train_i, :-1])/interval_frame_num

            name_head = img_dir + 'frame_' + str(img_num - length).zfill(6) + '.png'
            # print("\nWith label: head:", name_head)
            f_head = img_read_img_gradient(name_head,img_grad, crop, combined_image) - frame_0

            name_tail = img_dir + 'frame_' + str(img_num).zfill(6) + '.png'
            f_tail = img_read_img_gradient(name_tail,img_grad, crop, combined_image) - frame_0

            f_diff = f_tail - f_head

            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)

            y_train_list.append(label_head)
            for j in range(interval_frame_num-1):
                name_head = img_dir + 'frame_' + str(img_num + length * j).zfill(6) + '.png'
                # print("head:", name_head)

                f_head = img_read_img_gradient(name_head,img_grad, crop, combined_image) - frame_0

                name_tail = img_dir + 'frame_' + str(img_num + length * (j+1)).zfill(6) + '.png'
                f_tail = img_read_img_gradient(name_tail,img_grad, crop, combined_image) - frame_0

                f_diff = f_tail - f_head

                one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                    np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
                x_train_list.append(one_input_img)
                y_train_list.append(label_head + label_diff*(j+1))
    else:
        for train_i in range(1, len(df_merged)-test_sample-1):
            f_head = img_read_img_gradient(img_dir + df_merged.loc[train_i-1, "file"],img_grad, crop, combined_image) - frame_0

            f_tail = img_read_img_gradient(img_dir + df_merged.loc[train_i, "file"],img_grad, crop, combined_image) - frame_0
            # f_tail = (f_tail - np.min(f_tail)) / (np.max(f_tail) - np.min(f_tail))

            f_diff = f_tail - f_head
            # f_diff = (f_diff - np.min(f_diff)) / (np.max(f_diff) - np.min(f_diff))

            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                                np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)
            y_train_list.append(df_merged.iloc[train_i, :-1])

    x_train = np.array(x_train_list)
    print("dp: x.shape:",x_train.shape)
    if img_grad:
        x_train = np.transpose(x_train,[0,2,3,1])
        print("dp:x.shape(transpose)", x_train.shape)
    else:
        # x_train = np.squeeze(x_train, axis=4)
        x_train = np.transpose(x_train, [0, 2, 3, 1])
        print("dp:x.shape(transpose)", x_train.shape)
    y_train = np.array(y_train_list)
    return x_train, y_train, x_test, y_test

def get_feature_all_data(df_merged, img_dir, interval_frame,img_grad, crop, combined_image):
    x_train_list = []
    y_train_list = []
    frame_0 = img_read_img_gradient(img_dir + df_merged.loc[0, "file"], img_grad, crop, combined_image)
    print("Initial image_shape:", frame_0.shape)

    if interval_frame:
        interval_frame_num = interval_frame
        length = int(
            (int(df_merged.loc[1, "file"][-10:-4]) - int(df_merged.loc[0, "file"][-10:-4])) / interval_frame_num)
        print("length:", length)

        for train_i in range(1, len(df_merged) - 1):
            img_num = int(df_merged.loc[train_i, "file"][-10:-4])
            label_head = df_merged.iloc[train_i, :-1]
            label_diff = (df_merged.iloc[train_i + 1, :-1] - df_merged.iloc[train_i, :-1]) / interval_frame_num

            name_head = img_dir + 'frame_' + str(img_num - length).zfill(6) + '.png'
            f_head = img_read_img_gradient(name_head, img_grad, crop, combined_image) - frame_0

            name_tail = img_dir + 'frame_' + str(img_num).zfill(6) + '.png'
            f_tail = img_read_img_gradient(name_tail, img_grad, crop, combined_image) - frame_0

            f_diff = f_tail - f_head

            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                    np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)

            y_train_list.append(label_head)

            for j in range(interval_frame_num - 1):
                name_head = img_dir + 'frame_' + str(img_num + length * j).zfill(6) + '.png'
                f_head = img_read_img_gradient(name_head, img_grad, crop, combined_image) - frame_0
                # f_head = (f_head - np.min(f_head)) / (np.max(f_head) - np.min(f_head))

                name_tail = img_dir + 'frame_' + str(img_num + length * (j + 1)).zfill(6) + '.png'
                f_tail = img_read_img_gradient(name_tail, img_grad, crop, combined_image) - frame_0
                # f_tail = (f_tail - np.min(f_tail)) / (np.max(f_tail) - np.min(f_tail))

                f_diff = f_tail - f_head
                # f_diff = (f_diff - np.min(f_diff)) / (np.max(f_diff) - np.min(f_diff))

                one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                        np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
                x_train_list.append(one_input_img)

                y_train_list.append(label_head + label_diff * (j + 1))
    else:
        for train_i in range(1, len(df_merged)):
            f_head = img_read_img_gradient(img_dir + df_merged.loc[train_i - 1, "file"],img_grad, crop, combined_image) - frame_0
            # f_head = (f_head - np.min(f_head))/(np.max(f_head) - np.min(f_head))

            f_tail = img_read_img_gradient(img_dir + df_merged.loc[train_i, "file"],img_grad, crop, combined_image)  - frame_0
            # f_tail = (f_tail - np.min(f_tail))/(np.max(f_tail) - np.min(f_tail))

            f_diff = f_tail - f_head
            # f_diff = (f_diff - np.min(f_diff))/(np.max(f_diff) - np.min(f_diff))
            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                    np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)

            y_train_list.append(df_merged.iloc[train_i, :-1])

    x_train = np.array(x_train_list)
    # if img_grad:
    # x_train = np.transpose(x_train, [0, 2, 3, 1])
    # else:
    #     x_train = np.squeeze(x_train, axis=4)
    #     x_train = np.transpose(x_train, [0, 2, 3, 1])
    y_train = np.array(y_train_list)
    return x_train, y_train

def img_read(path):
    """
    read image
    :param path:  image_directory
    :return: image (array like )
    """
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (112, 112))
    img = img_to_array(img)
    img = input_norm(img)
    return img

def crop_img(img):
    x,y = img.shape;
    x = int(x*2/3)  #wigth
    y = int(y/2)    #heigth
    img = img[:x, :y]
    return img

def img_read_img_gradient(path=None,img_grad=True, crop = False,combined_image = False):
    """
    read image
    :param path:  image_directory
    :return: image (array like )
    """
    if img_grad:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (112, 112))
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  #(224,224)
        # sobelx = cv2.equalizeHist(sobelx)
        # sobelx = sobelx/np.max(np.abs(sobelx))
        # sobelx = (sobelx - np.min(sobelx)) / (np.max(sobelx) - np.min(sobelx))
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        # sobely = cv2.equalizeHist(sobely)
        # sobely = sobely / np.max(sobely)
        # img = np.concatenate([sobelx,sobely],axis=1)
        if combined_image:
            img = np.concatenate([sobelx,sobely],axis = 1)
        else:
            img = sobelx
    else:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (112, 112))
        img = img_to_array(img)
        img = np.squeeze(img,axis=2)  #(224,224)

    if crop:
        img = crop_img(img)
    return img


################################################# abandoned code ######################
#
# class subject(object):
#     def __init__(self, feature_path, label_path, img_path, group_size, interval=-1):
#         self.feature_path = feature_path
#         self.label_path = label_path
#         self.img_path = img_path
#         self.group_size = group_size
#         self.interval = interval
#
#     # no interval images
#     def data_loading(self):
#         """
#         load data
#         observation_value = None: return all six label values
#         observation = {0,1,2,3,4,5} --> {'rotX', 'rotY', 'rotZ', 'transX', 'transY', 'transZ'}
#         get x and y for the general use
#         :return: X and Y;
#         X: X.shape:(nums_of_instance,img_row,img_col,channel)
#         Y: Y.shape:(nums_of_instance,1)
#         """
#         data = pd.read_csv(self.label_path, delimiter=',')
#         # print("\nlen_data:{}\n".format(np.shape(data)))
#         pic_name = pd.read_csv(self.feature_path, delimiter=',')
#         # print("\nlen_pic_name:{}\n".format(np.shape(pic_name)))
#         # modify data:
#         if not np.any(pic_name.time_sec == 0):
#             pic_name['time_sec'] = round(pic_name['time_sec'], 2)  # match label_time_sec
#         pic_name['file'] = list(map(lambda x: x[-16:], pic_name['file']))  # modify img_name
#
#         # classify the label:drop time == 0 when it is not the threshold, and drop time_sec column for label
#         if np.any(data.iloc[0, :-1] != 0):
#             data = data.iloc[1:, :].reset_index(drop=True)
#
#         # drop rows of label data that time is ahead of pic_ time
#         if pic_name.loc[0, "time_sec"] > 0:
#             data = data[data["time_sec"] > pic_name.loc[0, "time_sec"]].reset_index(drop=True)
#
#         label = np.array(data.iloc[:, :-1])
#         self.time = np.array(data["time_sec"]).reshape(-1, 1)
#         # because the first frame may not be (0,0,0,0,0,0), therefore, we make each row to be deducted by first row
#         # then the first row is (0,0,0,0,0,0)
#         # after that, we add back the first row value to prediction
#         Y = label - label[0]
#         Y = np.concatenate((Y, self.time), axis=1)
#
#         # get feature
#         new_df = pd.merge(pic_name, data, how='right', on='time_sec')
#         pic_list = []
#
#         # print("THIS is img_path checkpoint:",self.img_path)
#         for i in range(len(new_df)):
#             name = new_df.loc[i, "file"]
#             # print("DP_line63",name)
#             path = self.img_path + name
#             # print("DP_line65", path)
#             # print("checkpoint2: path of frame:",path)
#             img = img_read(path)
#             pic_list.append(img)
#         X = np.array(pic_list).reshape(-1, img.shape[0], img.shape[1], img.shape[2])
#         return X, Y
#
#     # with interval images
#     def data_loading_with_interval(self, interval_frame):
#         """
#         load data
#         observation_value = None: return all six label values
#         observation = {0,1,2,3,4,5} --> {'rotX', 'rotY', 'rotZ', 'transX', 'transY', 'transZ'}
#         get x and y for the general use
#         :return: X and Y;
#         X: X.shape:(nums_of_instance,img_row,img_col,channel)
#         Y: Y.shape:(nums_of_instance,1)
#         """
#         data = pd.read_csv(self.label_path, delimiter=',')
#         print("\nlen_data:{}\n".format(np.shape(data)))
#         pic_name = pd.read_csv(self.feature_path, delimiter=',')
#         print("\nlen_pic_name:{}\n".format(np.shape(pic_name)))
#         # modify data:
#         if not np.any(pic_name.time_sec == 0):
#             pic_name['time_sec'] = round(pic_name['time_sec'], 2)  # match label_time_sec
#         pic_name['file'] = list(map(lambda x: x[-16:], pic_name['file']))  # modify img_name
#
#         # classify the label:drop time == 0 when it is not the threshold, and drop time_sec column for label
#         if np.any(data.iloc[0, :-1] != 0):
#             data = data.iloc[1:, :].reset_index(drop=True)
#
#         # drop rows of label data that time is ahead of pic_ time
#         if pic_name.loc[0, "time_sec"] > 0:
#             data = data[data["time_sec"] > pic_name.loc[0, "time_sec"]].reset_index(drop=True)
#
#         label = np.array(data.iloc[:, :-1])
#         self.time = np.array(data["time_sec"]).reshape(-1, 1)
#         # because the first frame may not be (0,0,0,0,0,0), therefore, we make each row to be deducted by first row
#         # then the first row is (0,0,0,0,0,0)
#         # after that, we add back the first row value to prediction
#         Y = label - label[0]
#         Y = np.concatenate((Y, self.time), axis=1)[1:]
#
#         # get feature
#         new_df = pd.merge(pic_name, data, how='right', on='time_sec')
#         pic_list = []
#
#         # get the interval name:
#         length = int((int(new_df.loc[1, "file"][-10:-4]) - int(new_df.loc[0, "file"][-10:-4])) / interval_frame)
#
#         # print("THIS is img_path checkpoint:",self.img_path)
#         # 5 frames : fn-1 - f0, ff1 - f0, ff2 - f0, ff3 - f0, fn - f0 (set interval_frame = 3)!!!!
#
#         # initail frame(standard)
#         frame_0 = img_read(self.img_path + new_df.loc[0, "file"])
#
#         for i in range(len(new_df) - 1):
#             batch_x = []
#             time_ = int(new_df.loc[i, "file"][-10:-4])
#             # adding fn-1 - f_0
#             f_head = img_read(self.img_path + new_df.loc[i, "file"]) - frame_0
#             print()
#             batch_x.append(input_norm(f_head))
#             # adding intermediate frames
#             for j in range(1, interval_frame + 1):
#                 name = self.img_path + 'frame_' + str(time_ + length * j).zfill(6) + '.png'
#                 img = img_read(name) - frame_0
#                 img = input_norm(img)
#                 batch_x.append(img)
#             # adding fn - f_0
#             f_end = img_read(self.img_path + new_df.loc[i + 1, "file"]) - frame_0
#             batch_x.append(input_norm(f_end))
#             if i == 0:
#                 print("\nstandard image_0:\n", str(f_end))
#                 print("\nafter norm\n", str(batch_x[4]))
#             # save each group into list
#             pic_list.append(batch_x)
#         X = np.array(pic_list)
#         # squeeze the color tunnel = 1, x.shape: (322,5,224,224)
#         X = np.squeeze(X, axis=4)
#         return X, Y
#
#     def get_data(self):
#         # data preprocessing
#         if self.interval == -1:
#             # without intermediate frames
#             x, y = self.data_loading()
#             x, y = group_frames_diff(x, y, group_size=self.group_size)  # fn-1 - f0, fn - f0
#         else:
#             # with intermediate frames
#             x, y = self.data_loading_with_interval(interval_frame=self.interval)
#         print("\n\nthe length of x and y after grouping:\n{}\n{}\n\n".format(x.shape, y.shape))
#         # input x/255 is wrong, but it is not a image.but we use two image difference,therefore, it is a small number!!!!
#
#         return x, y
#
#
# def input_norm(arr):
#     """
#     tranform arr to (0,1)
#     :param arr: shape: (224,224,1)
#     :return: normalized_arr = (1-0)*(arr-amin(arr)/amax(arr)-amin(arr)) + 0
#     """
#     # normalized_arr = (1 - 0) * (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr) + 1) + 0
#     normalized_arr = arr/255
#     return normalized_arr
#
#
# def tanh(x):
#     return np.tanh(x)
#
#
# def tanh_back(x):
#     vfunc = np.vectorize(lambda x: 1 / 2 * np.log((1 + x) / (1 - x)))
#     x_ori = vfunc(x)
#     return x_ori
#
#
# def group_frames_diff(x, y, group_size):
#     """
#     input:  x: array(frames,img_row,img_col,size)
#             y: array(frames,1)
#
#     output: x_batch:(None,img_row, img_col,size)
#             y_batch:(None,1)
#             size(y_batch) = size(x_batch)
#
#     set size =2 :input will be (fn-1-f0, fn-f0)to predict label ln;  total 2 frames as a batch,next group
#                  is (fn-f0,fn+1 - f0), a time series features
#     """
#     x_diff = x[1:] - x[0]
#     x_group = []
#     if group_size != 1:
#         y_group = y[group_size:]
#         for i in range(len(x_diff) - group_size + 1):
#             feature = []
#             for k in range(i, i + group_size):
#                 feature.append(x_diff[k])
#             feature = np.array(feature).reshape(-1, x.shape[1], x.shape[2])
#             x_group.append(feature)
#     else:
#         y_group = y[1:]
#         x_group = x_diff
#     return np.array(x_group).reshape(len(y_group), -1, x.shape[1], x.shape[2]), np.array(y_group).reshape(-1, 7)
#
# def split_train_random_test(x, y, test_sample, group_size):
#     """
#     Note: since the features in the batch exist in its prior and post batches, so it may lead to testing features trained for the model.
#     Thus, I delete the prior and post batch of each test data
#     :param x:
#     :param y:
#     :param test_sample: randomly select the number of the test data
#     :return: test and training data
#     """
#     total_index = list(range(len(y)))
#     test_index = random.sample(total_index, test_sample)
#     print("\nthe index of test data:\n", test_index)
#
#     array_test = np.array(test_index)
#     del_index = np.unique(np.concatenate((array_test - group_size + 1, array_test, array_test + group_size - 1)))
#     del_index = del_index[del_index >= 0]
#     del_index = del_index[del_index < len(y)]
#     train_index = np.delete(np.arange(len(y)), del_index)
#     print("\nthe index of train data:\n", train_index)
#
#     x_train, y_train = x[train_index], y[train_index]
#     x_test, y_test = x[test_index], y[test_index]
#
#     return x_train, y_train, x_test, y_test
#
#
# def split_train_test(x, y, test_sample):
#     """
#     Note: since the features in the batch exist in its prior and post batches, so it may lead to testing features trained for the model.
#     Thus, I delete the prior and post batch of each test data
#     :param x:
#     :param y:
#     :param test_sample: randomly select the number of the test data
#     :return: test and training data
#     """
#     total_index = list(range(len(y)))
#     test_index = total_index[-test_sample:]
#     train_index = total_index[:-test_sample]
#     x_train, y_train = x[train_index], y[train_index]
#     x_test, y_test = x[test_index], y[test_index]
#
#     return x_train, y_train, x_test, y_test
#
#
# def data_load_4subjects(group_size=2, test_subject=None, test_subject_list=None):
#     assert (test_subject in test_subject_list)
#     # print("test_subject_list",test_subject_list)
#     # print("test_subject",test_subject)
#     # for training data
#     x_train = []
#     y_train = []
#     x_train_dict = {}
#     y_train_dict = {}
#
#     for subject_id in test_subject_list:
#         print("This is load subject {}.\n".format(subject_id))
#         # if not os.path.exists(save_img_path+"{}/".format(subject_id)):
#         #     m = os.umask(0)
#         #     os.makedirs(save_img_path+"{}/".format(subject_id))
#         #     os.umask(m)
#
#         path = "/home/huiyuan/summer_project/gpfs/data/epilepsy/mri/hpardoe/head_pose_datasets/sub-NC{}_head_pose_data_zhao/".format(
#             subject_id)
#         # os.listdir: returns a list containing the names of the entries
#         label_name = [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
#         feature_name = [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
#         img_dir_name = [f for f in os.listdir(path) if f.endswith('cropped_frames')][0]
#         print("img_dir_name", img_dir_name)
#
#         feature_path = path + feature_name
#         label_path = path + label_name
#         img_path = path + img_dir_name + "/"
#
#         Subject = subject(feature_path=feature_path, label_path=label_path, img_path=img_path, group_size=group_size)
#         if subject_id == test_subject:
#             print("This is to load testing subject {}.\n".format(test_subject))
#             x_test, y_test = Subject.get_data()
#             print("len(x_test):{}\nlen(y_test):{}\n".format(len(x_test), len(y_test)))
#         else:
#             x, y = Subject.get_data()
#             if len(x_train) == 0:
#                 x_train = x
#                 y_train = y
#             else:
#                 x_train = np.concatenate((x_train, x), axis=0)
#                 y_train = np.concatenate((y_train, y), axis=0)
#             x_train_dict[subject_id] = x
#             y_train_dict[subject_id] = y
#     print("DP_line269_test:shape:\n{0}\n{1}\n{2}\n{3}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
#
#     return x_train, y_train, x_test, y_test, x_train_dict, y_train_dict
#
#
# def data_load_single_subject(test_subject=150, test_sample=80, group_size=2, interval=-1):
#     path = "/home/huiyuan/summer_project/gpfs/data/epilepsy/mri/hpardoe/head_pose_datasets/sub-NC{}_head_pose_data_zhao/".format(
#         test_subject)
#
#     x, y = get_data(path=path, interval_frame=False)
#     print("ordinary data:\n{}\n{}".format(x.shape, y.shape))
#     print('completed getting ordinary data')
#     x_train, y_train, x_test, y_test = split_train_test(x, y, test_sample=test_sample)
#     print("ordinary data:\n{}\n{}".format(x_train.shape, y_train.shape))
#     print('completed spliting training and testing data')
#     # selet 100-180th groups
#     # x_train_part, y_train_part = x_train[:], y_train[:]
#     return x_train, y_train, x_test, y_test
#
#
# def get_data(path, interval_frame):
#     # get data path
#     # path = "./sub-NC{}_head_pose_data_zhao/".format(test_subject)
#
#     # get and clean csv file:
#     data, pic_name, img_dir_name = clean_data(path)
#
#     # get feature and label
#     X, Y = get_feature(data, pic_name, img_dir_name, interval_frame)
#     #    pd.savetxt("feature.csv", X, delimiter=',',fmt='%.06f')
#     #    np.savetxt("label.csv", Y, delimiter=',',fmt='%.06f')
#     return X, Y
#
#
# def clean_data(path):
#     # get label, image_name and image_directory
#     label_name = path + [f for f in os.listdir(path) if f.endswith('head_pose.csv')][0]
#     feature_name = path + [f for f in os.listdir(path) if f.endswith('frame_times.csv')][0]
#     img_dir_name = path + [f for f in os.listdir(path) if f.endswith('cropped_frames')][0] + '/'
#
#     # load csv file
#     data = pd.read_csv(label_name, delimiter=',')
#     pic_name = pd.read_csv(feature_name, delimiter=',')
#
#     # match time between frame_times.csv and head_pose.csv
#     pic_name['time_sec'] = round(pic_name['time_sec'], 2)
#     pic_name['file'] = list(map(lambda x: x[-16:], pic_name['file']))
#
#     # drop rows when the time in frame_time.csv cannot find in head_pose
#     data = data[data["time_sec"] > pic_name.loc[0, "time_sec"]].reset_index(drop=True)
#     return data, pic_name, img_dir_name
#
#
# def img_read(path):
#     """
#     read image
#     :param path:  image_directory
#     :return: image (array like )
#     """
#     img = cv2.imread(path, 0)
#     img = cv2.resize(img, (224, 224))
#     img = img_to_array(img)
#     img = input_norm(img)
#     # img = MaxPooling(img, (4, 4))
#     # img = MaxPooling(img, (4, 4))
#     # img = input_norm(img)
#     return img
#
# def get_feature(data, pic_name, img_dir_name, interval_frame):
#     """
#
#     :param data: head_pose.csv
#     :param pic_name: frame_times.csv
#     :param img_dir_name: path of directory to save images
#     :param interval_frame: Boolean type: if False, then without intermediate frames;else the number of interval frames
#     :return: X: feature
#     """
#     if np.any(data.iloc[0, :-1] != 0):
#         label = data.iloc[:, :-1].to_numpy()
#         time = np.array(data["time_sec"]).reshape(-1, 1)
#         Label = label - label[0]
#         Label = np.concatenate((Label, time), axis=1)
#     else:
#         Label = np.array(data)
#
#     # merge two csv file to get image:
#     new_df = pd.merge(pic_name, data, how='right', on='time_sec')
#     pic_list = []
#
#     # initial frame(threshold)
#     frame_0 = img_read(img_dir_name + new_df.loc[0, "file"])
#     print("image_shape:", frame_0.shape)
#
#     if interval_frame != False and isinstance(interval_frame, int):
#         # set interval_frame = 3:(f0-f0,ff1-f0,ff2-f0,ff3-f0,f1-f0) -->label 1
#         length = int((int(new_df.loc[1, "file"][-10:-4]) - int(new_df.loc[0, "file"][-10:-4])) / interval_frame)
#         for i in range(len(new_df) - 1):
#             batch_x = []
#             img_num = int(new_df.loc[i, "file"][-10:-4])
#             # adding fn-1 - f_0
#             f_head = img_read(img_dir_name + new_df.loc[i, "file"]) - frame_0
#             batch_x.append(f_head)
#             # adding intermediate frames
#             for j in range(1, interval_frame + 1):
#                 name = img_dir_name + 'frame_' + str(img_num + length * j).zfill(6) + '.png'
#                 img = img_read(name) - frame_0
#                 batch_x.append(img)
#             # adding fn - f_0
#             f_end = img_read(img_dir_name + new_df.loc[i + 1, "file"]) - frame_0
#             batch_x.append(f_end)
#             # save each group into list
#             pic_list.append(batch_x)
#         X = np.array(pic_list)
#         # squeeze the color tunnel = 1, x.shape: (322,5,224,224)
#         X = np.squeeze(X, axis=4)
#         Y = Label[:-1]
#     else:
#         # save frame_0 and one frame:eg: (f0,f1),(f0,f2)... -->label 1, label 2 ...
#         for i in range(len(new_df) - 1):
#             batch_x = []
#             time_ = int(new_df.loc[i, "file"][-10:-4])
#             # adding fn-1 - f_0
#             f_head = img_read(img_dir_name + new_df.loc[i, "file"]) - frame_0
#             print()
#             batch_x.append(input_norm(f_head))
#             # adding intermediate frames
#             for j in range(1, interval_frame + 1):
#                 name = img_dir_name + 'frame_' + str(time_ + length * j).zfill(6) + '.png'
#                 img = img_read(name) - frame_0
#                 img = input_norm(img)
#                 batch_x.append(img)
#             # adding fn - f_0
#             f_end = img_read(img_dir_name + new_df.loc[i + 1, "file"]) - frame_0
#             batch_x.append(input_norm(f_end))
#             if i == 0:
#                 print("\nstandard image_0:\n", str(f_end))
#                 print("\nafter norm\n", str(batch_x[4]))
#             # save each group into list
#             pic_list.append(batch_x)
#         X = np.array(pic_list)
#         # squeeze the color tunnel = 1, x.shape: (322,5,224,224)
#         X = np.squeeze(X, axis=4)
#         Y = Label[1:]
#     return X, Y
#
#
# def MaxPooling(img, stride):
#     """
#     get feature of image by using maxpooling
#     :param img: each image
#     :param stride: steps to jump (4,4)
#     :return: img
#     """
#     M, N = img.shape
#     K, L = stride
#
#     MK = M // K
#     NL = N // L
#     try:
#         # split the matrix into 'quadrants'
#         Q1 = img[:MK * K, :NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))
#         Q2 = img[MK * K:, :NL * L].reshape(-1, NL, L).max(axis=2)
#         Q3 = img[:MK * K, NL * L:].reshape(MK, K, -1).max(axis=1)
#         Q4 = img[MK * K:, NL * L:].max()
#         # compose the individual quadrants into one new matrix
#         new_img = np.vstack([np.c_[Q1, Q3], np.c_[Q2, Q4]])
#     except:
#         new_img = img[:MK * K, :NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))
#     return new_img

