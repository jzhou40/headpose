#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:13:38 2019

@author: neil
"""
import sys
import datetime
import main_rotation
# sys.path.insert(1, "/u/erdos/students/hyuan11/MRI/upload")
# import time
# import test_video_regression_WithoutGT

currentDT = datetime.datetime.now()
YMD = "{}{}{}".format(currentDT.year, currentDT.month, currentDT.day)
HMS = "{}{}{}".format(currentDT.hour, currentDT.minute, currentDT.second)

if __name__ == '__main__':
    # while True:
    #     pass

    #############below is parameter###########
    """
    
    To run on Big purple:
    modify data_process.py : line 26: path : (path to load data)
    modify main_hui.py: line 73:save_dir_file : the path to save result 

    if need to do test only, change parameter Only_for_test = True
        To load model in test script, change parameter model_location (line60) 
    
    """
    # group_size = 3  # the number of frames as a group to predict observations set size = 2, means (fn-1 - f0, fn - f0) so x.shape = (-1,img_h,img_w,2)
    if sys.argv[1] == "noskip":
        interval_frame = 39
        #interval_frame = 39  # True or False
    elif sys.argv[1] == "skip":
        interval_frame = 1
    else:
        assert (int(sys.argv[1]) >= 1 and int(
            sys.argv[1]) <= 39), "Please enter a number larger than 1 and smaller than 39"
        interval_frame = int(sys.argv[1])

    if sys.argv[2] == "original":
        img_grad = False  # True or False
    elif sys.argv[2] == "gradient":
        img_grad = True

    # data_path = sys.argv[4] # load data
    save_path = sys.argv[3]  # save results

    try:
        judge_rot_trans = sys.argv[4]
    except:
        judge_rot_trans = "rot"
        # judge_rot_trans = 'rot'  # rot or trans

    LOOCV = False
    batch = int(sys.argv[5])  # 5, 10, 20,22, 110, 220
    epochs = int(sys.argv[6])
    if sys.argv[7] == "combined":   # True or False
        combined_image = True
    elif sys.argv[7] == "single":
        combined_image = False
    test_sample = 100
    withGT = True
    # img_grad = True  #sys.argv[2]
    subject = 'all'
    col = judge_rot_trans  # 'rot'  # "rot' or 'trans'

    ##############above is parameter##########

    # To make some variables based on parameters
    if not LOOCV:
        print("subject:{};  col:{}".format(subject, col))
        # assert col in ['rot', "trans"], "please enter rot or trans!"
        if subject != "all":
            pass
            # option:["4subject", 150, 232, 233, 234,...]
            # save_dir_file = "/u/erdos/students/hyuan11/MRI/upload/{}_{}_sub{}/".format(YMD, HMS, subject)
            # if save_path[-1] == "/":
            #     save_dir_file = save_path + "{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
            # else:
            #     save_dir_file = save_path + "/{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
            # judgement = 0
            # print("To use sub_{}".format(subject))
        else:
            subject_list = [232, 233, 234, 235, 236, 237,239, 240, 242, 243, 244, 245, 246,247,248, 249, 250, 251, 252, 254, 255] #for LOO test [150,232, 233, 234]
            #subject_list = [150]
            # save_dir_file = "/u/erdos/students/hyuan11/MRI/upload/{}_{}_multiple_subs/".format(YMD, HMS)
            if save_path[-1] == "/":
                save_dir_file = save_path + "{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
            else:
                save_dir_file = save_path + "/{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
            judgement = 1
            print("To use multiple subs")

        if col == "rotY":
            # columns_name = ['rotX', 'rotY', 'rotZ']
            columns_name = [col]
        elif col == "rot":
            columns_name = ['rotX', 'rotY', 'rotZ']
        else:
            columns_name = ["transX", "transY", "transZ"]
    else:
        # LOOCV
        train_subject_list = [150]  # ,233,234,235,236,237,240,242,243,244,245]
        test_subject_list = [232]
        if save_path[-1] == "/":
            save_dir_file = save_path + "Loocv_{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
        else:
            save_dir_file = save_path + "/Loocv_{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
        if col == "rot":
            columns_name = ['rotX', 'rotY', 'rotZ']
        else:
            columns_name = ["transX", "transY", "transZ"]

        judgement = 1
        print("To use multiple subs")

    # enter the functions with some choices, eg only test, train model, with interval_frames or not...
    # if Only_for_test:
    #     """
    #     judge_rot_trans; string--> 'rot' or 'trans'
    #     """
    #     print("Only for test")
    #     test_video_regression_WithoutGT.main_test_NoGT(judge_rot_trans = judge_rot_trans,withGT = withGT,
    #                                                    interval_frame = interval_frame, img_grad = img_grad,
    #                                                    model_location = model_location,Whole_data = False,
    #                                                    save_path = save_path,test_sample=test_sample,
    #                                                     subject_list=subject_list)
    # test_regression_toCSV_copy()
    # else:

    if interval_frame:
        print("there are intermediate frames")
    else:
        print("there are NOT intermediate frames")

    if judgement == 0:
        main_rotation.rotation_single_sub(columns_name=columns_name, save_dir_file=save_dir_file, subject=subject,
                                          batch=batch, epochs=epochs,
                                          test_sample=test_sample, interval_frame=interval_frame)
    else:
        if not LOOCV:#this
            # multiple_subs with cutting train and test from one dataset this one
            main_rotation.rotation_multiple_subs(columns_name=columns_name, save_dir_file=save_dir_file,
                                                 subject_list=subject_list,
                                                 test_sample=test_sample, interval_frame=interval_frame, batch=batch,
                                                 epochs=epochs, img_grad=img_grad,combined_image = combined_image)  # , data_path = data_path)
        else:
            main_rotation.LOOCV(columns_name=columns_name, save_dir_file=save_dir_file,
                                train_subject_list=train_subject_list,
                                test_subject_list=test_subject_list,
                                interval_frame=interval_frame,
                                batch=batch, epochs=epochs)

    print("complete done!")