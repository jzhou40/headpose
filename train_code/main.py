#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:13:38 2019

@author: neil
"""
import sys
import datetime
import measurement

currentDT = datetime.datetime.now()
YMD = "{}{}{}".format(currentDT.year, currentDT.month, currentDT.day)
HMS = "{}{}{}".format(currentDT.hour, currentDT.minute, currentDT.second)

if __name__ == '__main__':
    if sys.argv[1] == "noskip":
        interval_frame = 39
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

    save_path = sys.argv[3]  # save results

    try:
        judge_rot_trans = sys.argv[4]
    except:
        judge_rot_trans = "rot"

    LOOCV = False
    batch = int(sys.argv[5])  # 5, 10, 20,22, 110, 220
    epochs = int(sys.argv[6])
    if sys.argv[7] == "combined":   # True or False
        combined_image = True
    elif sys.argv[7] == "single":
        combined_image = False
    test_sample = 100
    withGT = True
    subject = 'all'
    col = judge_rot_trans  # 'rot'  # "rot' or 'trans'

    ##############above is parameter##########

    # To make some variables based on parameters
    if not LOOCV:
        print("subject:{};  col:{}".format(subject, col))
        # assert col in ['rot', "trans"], "please enter rot or trans!"
        if subject != "all":
            pass
        else:
            subject_list = [232, 233, 234, 235, 236, 237,239, 240, 242, 243, 244, 245, 246,247,248, 249, 250, 251, 252, 254, 255] #for LOO test [150,232, 233, 234]
            
            if save_path[-1] == "/":
                save_dir_file = save_path + "{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
            else:
                save_dir_file = save_path + "/{}_{}_multiSubs_{}/".format(sys.argv[1], sys.argv[2], col)
            judgement = 1
            print("To use multiple subs")

        if col == "rotY":
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

    if interval_frame:
        print("there are intermediate frames")
    else:
        print("there are NOT intermediate frames")

    if judgement == 0:
        measurement.rotation_single_sub(columns_name=columns_name, save_dir_file=save_dir_file, subject=subject,
                                          batch=batch, epochs=epochs,
                                          test_sample=test_sample, interval_frame=interval_frame)
    else:
        if not LOOCV:#this
            # multiple_subs with cutting train and test from one dataset this one
            measurement.rotation_multiple_subs(columns_name=columns_name, save_dir_file=save_dir_file,
                                                 subject_list=subject_list,
                                                 test_sample=test_sample, interval_frame=interval_frame, batch=batch,
                                                 epochs=epochs, img_grad=img_grad,combined_image = combined_image)  # , data_path = data_path)
        else:
            measurement.LOOCV(columns_name=columns_name, save_dir_file=save_dir_file,
                                train_subject_list=train_subject_list,
                                test_subject_list=test_subject_list,
                                interval_frame=interval_frame,
                                batch=batch, epochs=epochs)

    print("complete done!")