#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_func(columns, sub, save_dir, label_train, label_test, pred_train, pred_val, training_loss, val_loss):

    columns = columns
    print("lenth of label and pred: {} ; {}".format(len(label_train['time_sec'])-1,len(pred_train['time_sec'])))
    for column in columns:
        plt.figure()
        plt.plot(np.array(label_train['time_sec'])[1:],np.array(label_train[column])[1:])
        plt.plot(np.array(pred_train['time_sec']),np.array(pred_train[column]))
        plt.legend(("ground_truth","prediction"))
        plt.title('{} training {}'.format(sub,column))
        plt.ylabel(column)
        plt.xlabel('time_sec')
        if columns == ['rotX', 'rotY', 'rotZ']:
            plt.ylim((-0.2, 0.2))
        else:
            plt.ylim((-4, 4))
        plt.savefig(save_dir +"training_{}_{}.png".format(sub,column))

        plt.figure()
        plt.plot(np.array(label_test['time_sec'])[1:],np.array(label_test[column])[1:])
        plt.plot(np.array(pred_val['time_sec']),np.array(pred_val[column]))
        plt.title('{} testing {}'.format(sub,column))
        plt.legend(("ground_truth","prediction"))
        plt.ylabel(column)
        plt.xlabel('time_sec')
        if columns == ['rotX', 'rotY', 'rotZ']:
            plt.ylim((-0.2, 0.2))
        else:
            plt.ylim((-4, 4))
        plt.savefig(save_dir + 'testing_{}_{}.png'.format(sub,column))

    plt.figure()
    #plot from 10th epochs since the first 10 scale is too large compared to the later ones.
    plt.plot(training_loss[10:])
    plt.plot(val_loss[10:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_dir + "learning_curve_{}.png".format(sub))


