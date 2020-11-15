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
    path = "/u/erdos/csga/jzhou40/MRI_Project/sub-NC{}_head_pose_data_zhao/".format(test_subject)
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
    for test_i in reversed(range(-1, -test_sample - 1, -1)) :

        if test_i == -test_sample :
            print("dp:test_img_head:{}".format(df_merged.loc[len(df_merged) + test_i - 1, "file"]))
            print("dp:test_img_tail:{}\n".format(df_merged.loc[len(df_merged) + test_i, "file"]))

        test_img_num = int(df_merged.loc[len(df_merged) + test_i, "file"][-10 :-4])
        test_i_name = img_dir + 'frame_' + str(test_img_num).zfill(6) + '.png'
        test_i_1_name = img_dir + 'frame_' + str(test_img_num - 1).zfill(6) + '.png'

        f_head = img_read_img_gradient(test_i_1_name, img_grad, crop, combined_image) - frame_0
        f_tail = img_read_img_gradient(test_i_name, img_grad, crop, combined_image) - frame_0
        f_diff = f_tail - f_head

        one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
        x_test_list.append(one_input_img)

        y_test_list.append(df_merged.iloc[len(df_merged) + test_i, :-1])

    x_test = np.array(x_test_list)
    print("dp:x_test.shape:",x_test.shape)
    if img_grad:
        x_test = np.transpose(x_test,[0,2,3,1])
        print("dp:x_test.shape", x_test.shape)
    else:
        x_test = np.transpose(x_test, [0, 2, 3, 1])
        print("dp:x_test.shape", x_test.shape)

    y_test = np.array(y_test_list)
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

            f_diff = f_tail - f_head

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

                name_tail = img_dir + 'frame_' + str(img_num + length * (j + 1)).zfill(6) + '.png'
                f_tail = img_read_img_gradient(name_tail, img_grad, crop, combined_image) - frame_0

                f_diff = f_tail - f_head

                one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                        np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
                x_train_list.append(one_input_img)

                y_train_list.append(label_head + label_diff * (j + 1))
    else:
        for train_i in range(1, len(df_merged)):
            f_head = img_read_img_gradient(img_dir + df_merged.loc[train_i - 1, "file"],img_grad, crop, combined_image) - frame_0

            f_tail = img_read_img_gradient(img_dir + df_merged.loc[train_i, "file"],img_grad, crop, combined_image)  - frame_0

            f_diff = f_tail - f_head
            one_input_img = ([f_head, f_tail, f_diff] - np.min([f_head, f_tail, f_diff])) / (
                    np.max([f_head, f_tail, f_diff]) - np.min([f_head, f_tail, f_diff]))
            x_train_list.append(one_input_img)

            y_train_list.append(df_merged.iloc[train_i, :-1])

    x_train = np.array(x_train_list)
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
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
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

