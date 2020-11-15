    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:52:59 2019

@author: neil
"""
from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1020)
import os
import random
from keras.models import Sequential
from keras.layers.convolutional import  MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Lambda, Activation, RepeatVector, Dot, Concatenate
from keras.preprocessing.image import img_to_array
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam,Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.callbacks import Callback
import keras.backend as K
from keras import regularizers
from keras import Input, Model
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16

def generator_data(x_dic,y_dic,random_list):
    while True:
        for index_list in random_list:
            x_batch = []
            y_batch = []
            for key in x_dic.keys():
                x_batch.append(x_dic[key][index_list])
                y_batch.append(y_dic[key][index_list])
            x_batch = np.array(x_batch).reshape(-1, x_dic[key].shape[1], x_dic[key].shape[2],x_dic[key].shape[3])  # 10*13
            y_batch = np.array(y_batch).reshape(-1, y_dic[key].shape[1])
            yield x_batch, y_batch

def r_squared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def weighted_MSE_rot(y_true, y_pred):
    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*200*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    return K.mean(K.square(y_pred -y_true)*y_weight, axis=-1)

def weighted_MSE_trans(y_true, y_pred):
    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*10*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    return K.mean(K.square(y_pred -y_true)*y_weight, axis=-1)
######################## basic cnn model  ########################
def get_model_structure(input_shape, output_column):
    """
    :param xtrain:(group_numbers, img_height,img_weight,group_size)
    :param ytrain:shape(-1,1)
    :param xVal:validation dataset
    :param yVal:validation dataset
    :return:save model into self, no returns
    """
    # define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',input_shape=input_shape))  # input is a greyscale pic
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(64, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(len(output_column), activity_regularizer=regularizers.l2(0.01),kernel_regularizer=regularizers.l2(0.01)))
    if output_column == ["rotX", "rotY", "rotZ"]:
        cnn.compile(optimizer=Adam(lr=0.001), metrics=["mse"], loss = weighted_MSE_rot) # weighted_MSE_rot
    elif output_column == ["transX", "transY", "transZ"]:
        cnn.compile(optimizer=Nadam(lr=0.001), metrics=["mse"], loss = weighted_MSE_trans)
    print(cnn.summary())

    return cnn

def model_fitting_generator(model, batch_size, epochs, x_train, y_train, x_val, y_val, save_dir_file, ins_per_sub):#,sub233):
    """

    :param model: model construction
    :param batch_size:
    :param epochs:
    :param x_train: dict, key = sub name; value is ndarray
    :param y_train: dict
    :param x_val: ndarry, last 100 frames of each sub with real values concatenate
    :param y_val: ndarry
    :return:
    """
    ckpoint = ModelCheckpoint(save_dir_file + 'weights_single_best.h5', #"weights-improvement-epoch_{epoch:02d}.h5"
                              monitor='val_loss', verbose=1, save_best_only=True,
                              mode='min') # every model
    es = EarlyStopping(min_delta=0.0001, patience=100, mode='min', monitor='val_loss', restore_best_weights=True,
                       verbose=1)
    rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100, verbose=1, min_lr=0.0001)

    # use GPU
    gpu_no = 0


    random_list = np.arange(ins_per_sub)
    np.random.shuffle(random_list)
    sample_per_sub =  batch_size # 5, 10, 20,22, 110, 220
    print("steps_per_epoch:", ins_per_sub // sample_per_sub)
    random_list = random_list.reshape(-1, sample_per_sub)  # 860*10

    with tf.device("/gpu:" + str(gpu_no)):
        print("this is to run gpu")
        History = model.fit_generator(generator_data(x_dic = x_train,y_dic = y_train,random_list = random_list),
                                    steps_per_epoch= ins_per_sub//sample_per_sub, #ins_per_sub//batch_size,
                                    epochs=epochs,
                                    validation_data=(x_val, y_val),
                                    callbacks=[es, rp,ckpoint],
                                    verbose=2,
                                    shuffle=True)

    training_loss = History.history["loss"]
    val_loss = History.history["val_loss"]
    np.savetxt(save_dir_file + "training_loss.csv", np.array(training_loss))
    np.savetxt(save_dir_file + "val_loss.csv", np.array(val_loss))

    return model, training_loss, val_loss

