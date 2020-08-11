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

# import matplotlib.pyplot as plt
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
            # generator_data_inner(x_dic = x_dic,y_dic = y_dic,index_list=random_list[i])
            x_batch = []
            y_batch = []
            for key in x_dic.keys():
                x_batch.append(x_dic[key][index_list])
                y_batch.append(y_dic[key][index_list])
            x_batch = np.array(x_batch).reshape(-1, x_dic[key].shape[1], x_dic[key].shape[2],x_dic[key].shape[3])  # 10*13
            y_batch = np.array(y_batch).reshape(-1, y_dic[key].shape[1])
            yield x_batch, y_batch


######################## various metrics  ########################
# def r_squared(y_true, y_pred):
#     #means of six R squared values
#     a = K.sum(K.square(y_pred - K.mean(y_true)), axis = 0, keepdims = True)
#     b = K.sum(K.square(y_true - K.mean(y_true)), axis = 0, keepdims = True)
#     return K.mean(a/b)

def r_squared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# def corr(y_true,y_pred):
#     """
#     input: shape = (n,6)
#     """
#     nominator = K.sum((y_true-K.mean(y_true,axis=0,keepdims = True))*(y_pred-K.mean(y_pred,axis = 0,keepdims = True)), axis=0, keepdims=True)
#     denominator = K.std(y_true, axis=0, keepdims=True)*K.std(y_pred, axis=0, keepdims=True)
#     pcc = K.sum(nominator/denominator)
#     return pcc
#
# def MSE(y_true, y_pred):
#     mse = K.mean(K.sum(K.square(y_true - y_pred), axis = 1, keepdims = True),axis = 0)
#     return mse

#### MAE
# def weighted_MAE_rot(y_true, y_pred):
#     # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
#     # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
#     # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))
#
#     condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)
#     y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*10,tf.ones_like(y_true,dtype="float32"))
#     # y_weight = y_true*10000
#     return K.mean(K.abs((y_pred - y_true)*y_weight))  #modify： if <10^-4, *1; else 100  (233,234,244,245)
#
# def weighted_MAE_trans(y_true, y_pred):
#     # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
#     # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
#     # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))
#
#     condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
#     y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*100*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
#     # y_weight = y_true*10000
#     return K.mean(K.abs((y_pred - y_true)*y_weight))
####

#### MSE
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

def weighted_MSE_trans(y_true, y_pred):
    # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
    # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
    # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))

    condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
    y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
    # y_weight = y_true*10000
    return K.mean(K.square(y_pred -y_true)*y_weight, axis=-1)
####

####  RMSE
# def weighted_RMSE_rot(y_true, y_pred):
#     # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
#     # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
#     # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))
#
#     condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*0.05)
#     y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*10,tf.ones_like(y_true,dtype="float32"))
#     # y_weight = y_true*10000
#     return K.sqrt(K.mean(K.square(y_pred -y_true)*y_weight, axis=-1))
#
# def weighted_RMSE_trans(y_true, y_pred):
#     # return K.mean(K.abs((y_pred - y_true)*y_true*10000))
#     # condition = tf.greater(y_true*10000,tf.ones_like(y_true,dtype="float32"))
#     # y_weight = tf.where(condition,y_true*10000,tf.ones_like(y_true,dtype="float32"))
#
#     condition = tf.greater(tf.abs(y_true),tf.ones_like(y_true,dtype="float32")*1)
#     y_weight = tf.where(condition,tf.ones_like(y_true,dtype="float32")*100*tf.abs(y_true),tf.ones_like(y_true,dtype="float32"))
#     # y_weight = y_true*10000
#     return K.sqrt(K.mean(K.square(y_pred -y_true)*y_weight, axis=-1))
########################



# def loss_function(alpha = 0.5):
#     def loss_mse_corr(y_true, y_pred):
#         mse = MSE(y_true, y_pred)
#         diff_corr = corr(y_true, y_pred)
#         loss = mse + alpha*diff_corr
#         return loss
#     return loss_mse_corr


##########################  custom callbacks #####################

# class myCallback(Callback):
#     def __init__(self,sub233,save_dir_file):
#         self.sub233 = sub233
#         self.save_dir_file = save_dir_file
#     def on_epoch_end(self, epoch, logs={}):
#         pred_233 = self.model.predict(self.sub233)
#         np.savetxt(self.save_dir_file + "epoch_{}.csv".format(epoch), np.array(pred_233))
#
#
#

####################### pretrained model #########################
# def pretrain_model_vgg16(input_shape, output_column):
#     vgg16 = VGG16(include_top=False,weights='imagenet')
#
#     input = Input(shape=input_shape, name = 'image_input')
#     x = vgg16(input)
#     x = Flatten(name="flatten")(x)
#     x = Dense(2048, activation='relu')(x)
#     # x = BatchNormalization()(x)
#     # x = Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     # x = BatchNormalization()(x)
#     # x = Dropout(0.5)(x)
#     output = Dense(len(output_column), activity_regularizer=regularizers.l2(0.01),kernel_regularizer=regularizers.l2(0.01), name='predictions')(x)
#     model = Model(inputs=input, outputs=output)
#
#     if output_column == ["rotX", "rotY", "rotZ"]:
#         model.compile(optimizer=Nadam(lr=0.001), metrics=[r_squared,"mse","mae"], loss="mse") # weighted_MSE_rot
#     elif output_column == ["transX", "transY", "transZ"]:
#         model.compile(optimizer=Nadam(lr=0.001), metrics=[r_squared,"mse","mae"], loss="mean_squared_error") # weighted_MSE_trans
#
#     print(model.summary())
#     return model


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
    # cnn.add((MaxPooling2D(pool_size=(2, 2),input_shape=input_shape)))
    cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',input_shape=input_shape))  # input is a greyscale pic
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    # cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    # cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(BatchNormalization())

    #
    # cnn.add(Dense(512, activation='relu'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))

    # cnn.add(Dense(256, activation='relu'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))

    # cnn.add(Dense(256, activation='relu'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))
    #
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))
    #
    cnn.add(Dense(64, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(len(output_column), activity_regularizer=regularizers.l2(0.01),kernel_regularizer=regularizers.l2(0.01)))
    # cnn.compile(optimizer=Nadam(lr=0.001), metrics=[ "mean_squared_error"], loss='mean_absolute_error')
    if output_column == ["rotX", "rotY", "rotZ"]:
        cnn.compile(optimizer=Adam(lr=0.001), metrics=["mse"], loss = weighted_MSE_rot) # weighted_MSE_rot
    elif output_column == ["transX", "transY", "transZ"]:
        cnn.compile(optimizer=Nadam(lr=0.001), metrics=["mse"], loss = weighted_MSE_trans)
        # cnn.compile(optimizer=Nadam(lr=0.001), metrics=[r_squared,"mse","mae"], loss=weighted_MSE_trans) ##weighted_MSE_trans

    print(cnn.summary())

    return cnn

######################## cnn with attention model ########################
# def softmax(x):
#     return K.sigmoid(x,axis=-1)
#
# def one_step_attention_layer(h,X):
#     """
#
#     :param prob: input_shape: (m,1,1,512) probability of some pixel
#     :param X: input_shape: (m,1,1,512) a pixel with 512 dimension
#     :return: tensor with attention probability (1,1,1)
#     """
#
#
#     # Repeat vector to match a's dimensions
#     # h_repeat = RepeatVector(512)(X)
#     # Calculate attention weights
#     att = Concatenate([X, h],axis=-1)
#     att = Dense(8, activation="tanh")(att)
#     att = Dense(1, activation="relu")(att)
#     attention = Activation(softmax, name='attention_weights')(att)
#     # Calculate the tensor with attention
#     tensor = Dot([attention, X],axes=1)
#     return tensor
#
# def attention_layer(X=None):
#     """
#
#     :param X: Layer input (None, 14, 14,512)
#     :return:
#     """
#     outputs = []
#     h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0],1,1,K.shape(X)[-1])))(X)  #shape:m,14,14
#     # probability = tf.map_fn(lambda: one_step_attention_layer, X)
#     for i in range(14):
#         for j in range(14):
#             outputs.append(one_step_attention_layer(h,X[:,i,j,:]))
#     return outputs
#
# def get_model_structure_attention(input_shape, output_column):
#     """
#     :param xtrain:(group_numbers, img_height,img_weight,group_size)
#     :param ytrain:shape(-1,1)
#     :param xVal:validation dataset
#     :param yVal:validation dataset
#     :return:save model into self, no returns
#     """
#     # define CNN model
#
#     Input_layer = Input(shape=input_shape, name="input_shape")
#     cnn = Conv2D(64, kernel_size=3, activation='relu', padding='same',input_shape=input_shape)(Input_layer)  # input is a greyscale pic
#     cnn = BatchNormalization()(cnn)
#     cnn = Conv2D(64, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = Conv2D(128, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
#     cnn = Conv2D(256, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = Conv2D(256, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = Conv2D(256, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
#     cnn = Conv2D(512, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = Conv2D(512, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = Conv2D(512, kernel_size=3, activation='relu', padding='same')(cnn)
#     cnn = BatchNormalization()(cnn)
#     cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
#     attention_1 = attention_layer(X=cnn)      #14*14*512
#     cnn = Flatten()(attention_1)
#     output = Dense(512, activation='relu')(attention_1)
#     output = BatchNormalization()(output)
#     output = Dense(512, activation='relu')(output)
#     output = BatchNormalization()(output)
#     output = Dropout(0.5)(output)
#     output = Dense(256, activation='relu')(output)
#     output = BatchNormalization()(output)
#     output = Dropout(0.5)(output)
#     output = Dense(256, activation='relu')(output)
#     output = BatchNormalization()(output)
#     output = Dropout(0.5)(output)
#     output = Dense(128, activation='relu')(output)
#     output = BatchNormalization()(output)
#     output = Dropout(0.5)(output)
#     output = Dense(64, activation='relu')(output)
#     output = BatchNormalization()(output)
#     output = Dense(len(output_column), activity_regularizer=regularizers.l2(0.01),kernel_regularizer=regularizers.l2(0.01))(output)
#     model = Model(inputs=Input_layer, outputs=output)
#     # model.compile(optimizer=Nadam(lr=0.001), metrics=[ "mean_squared_error"], loss='mean_absolute_error')
#     if output_column == ["rotX", "rotY", "rotZ"]:
#         model.compile(optimizer=Nadam(lr=0.001), metrics=[r_squared,"mse","mae"], loss=weighted_MSE_rot)
#     elif output_column == ["transX", "transY", "transZ"]:
#         model.compile(optimizer=Nadam(lr=0.001), metrics=[r_squared,"mse","mae"], loss="mean_squared_error") ##weighted_MSE_trans
#     print(model.summary())
#
#     return model

########################   Model fiting ########################
# def model_fitting(model, batch_size=128, epochs=None, x_train = None, y_train = None, x_val = None, y_val = None,save_dir_file = None):
#
#     ckpoint = ModelCheckpoint(save_dir_file +'weights_single_best.h5', monitor='val_r_squared', verbose=1,save_best_only=True, mode='max')
#     es = EarlyStopping(min_delta=0.0001, patience=200, mode='min', monitor='val_loss', restore_best_weights=True,
#                        verbose=1)
#     rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100,verbose = 1,min_lr = 0.0001)
#
#     #use GPU
#     gpu_no = 0
#     with tf.device("/gpu:" + str(gpu_no)):
#         print("this is to run gpu")
#         if len(x_val)==0:
#             History = model.fit(x_train, y_train,
#                                 batch_size = batch_size,
#                                 epochs = epochs,
#                                 validation_split = 0.1,
#                                 callbacks =[es, rp, ckpoint],
#                                 verbose = 2)
#         else:
#             History = model.fit(x_train, y_train,
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 validation_data=(x_val, y_val),
#                                 callbacks=[es, rp, ckpoint],
#                                 verbose=2)
#
#         training_loss = History.history["loss"]
#         val_loss = History.history["val_loss"]
#         np.savetxt(save_dir_file + "training_loss.csv", np.array(training_loss))
#         np.savetxt(save_dir_file + "val_loss.csv", np.array(val_loss))
#
#     return model, training_loss, val_loss


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
        # History = model.fit_generator(generator_data(save_dir_file + "data_generator/"),
                                    steps_per_epoch= ins_per_sub//sample_per_sub, #ins_per_sub//batch_size,
                                    epochs=epochs,
                                    validation_data=(x_val, y_val),
                                    #callbacks=[es, rp,ckpoint,myCallback(sub233 = sub233,save_dir_file = save_dir_file)],
                                    callbacks=[es, rp,ckpoint],
                                    verbose=2,
                                    shuffle=True)

    training_loss = History.history["loss"]
    val_loss = History.history["val_loss"]
    np.savetxt(save_dir_file + "training_loss.csv", np.array(training_loss))
    np.savetxt(save_dir_file + "val_loss.csv", np.array(val_loss))

    return model, training_loss, val_loss



########################  comments   ################################################
# class cnn(object):
#     """
#     input image size:(img_row,img_col,1)
#     default times = 3
#     default epoc hs =1000
#     default batch_size = 156
#     """
#
#     def __init__(self, batch_size=156, epochs=1000,input_shape = None):
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.input_shape = input_shape
#         self.adam = Adam(lr = 0.001)
#         # self.class_weight = class_weight
#
#
#     def fitting(self, xtrain, ytrain, xVal= [], yVal = []):
#         """
#         :param xtrain:
#         :param ytrain:
#         :param xVal:
#         :param yVal:
#         :return:
#         """
#         # define CNN model
#
#         cnn = Sequential()
#         cnn.add(Conv2D(32, kernel_size=4, activation='relu', padding='same',
#                        input_shape=self.input_shape))  # input is a greyscale pic
#         cnn.add(BatchNormalization())
#         cnn.add(Conv2D(32, kernel_size=4, activation='relu', padding='same'))
#         cnn.add(BatchNormalization())
#         cnn.add(MaxPooling2D(pool_size=4))  # strides: If None, it will default to pool_size.
#         cnn.add(Conv2D(64, kernel_size=4, activation='relu', padding='same'))
#         cnn.add(BatchNormalization())
#         cnn.add(Conv2D(64, kernel_size=4, activation='relu', padding='same'))
#         cnn.add(BatchNormalization())
#         cnn.add(MaxPooling2D(pool_size=4))
#         cnn.add(Flatten())
#         cnn.add(Dense(256, activation='relu'))
#         cnn.add(BatchNormalization())
#         # cnn = Dropout(0.25)(cnn)
#         cnn.add(Dense(128, activation='relu'))
#         cnn.add(BatchNormalization())
#         cnn.add(Dense(64, activation='relu'))
#         cnn.add(BatchNormalization())
#         cnn.add(Dropout(0.25))
#         cnn.add(Dense(3, activity_regularizer=regularizers.l2(0.01)))
#         cnn.compile(optimizer=self.adam, metrics=[r_squared, "mean_squared_error"], loss="mean_absolute_error")
#         # print(cnn.summary())
#         self.model = cnn
#
#         es = EarlyStopping(min_delta=0.000001, patience=100, mode='min', monitor='val_loss',restore_best_weights = True,verbose = 2)
#         rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20,verbose = 2,min_lr = 0.0005)
#
#         if len(xVal) ==0 and len(yVal) == 0:
#             self.model.fit(xtrain, ytrain,
#                     batch_size=self.batch_size,
#                     epochs=self.epochs,
#                     validation_split = 0.3,
#                     callbacks=[es, rp],
#                     verbose = 2)
#         else:
#             self.model.fit(xtrain, ytrain,
#                     batch_size=self.batch_size,
#                     epochs=self.epochs,
#                     validation_data=(xVal, yVal),
#                     callbacks=[es, rp],
#                     verbose = 2)
#
#     def fitting_single_column_seq(self, xtrain, ytrain, xVal=[], yVal=[]):
#         """
#         :param xtrain:(group_numbers, img_height,img_weight,group_size)
#         :param ytrain:shape(-1,1)
#         :param xVal:validation dataset
#         :param yVal:validation dataset
#         :return:save model into self, no returns
#         """
#         # define CNN model
#
#         cnn = Sequential()
#         # cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',data_format = "channels_first",
#         #                input_shape=self.input_shape))  # input is a greyscale pic
#         # cnn.add(BatchNormalization())
#         # cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(MaxPooling2D(pool_size=(2, 2), data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Conv2D(128, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(MaxPooling2D(pool_size=(2, 2), data_format = "channels_first"))
#         # cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Conv2D(256, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(MaxPooling2D(pool_size=(2, 2), data_format = "channels_first"))
#         # cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Conv2D(512, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         # cnn.add(BatchNormalization())
#         # cnn.add(MaxPooling2D(pool_size=(2, 2), data_format = "channels_first"))
#         #######new add######
#         cnn.add(Conv2D(64, kernel_size=3, activation='relu', padding='same',data_format = "channels_first",
#                        input_shape=self.input_shape))  # input is a greyscale pic
#         cnn.add(BatchNormalization())
#         cnn.add(Conv2D(32, kernel_size=3, activation='relu', padding='same',data_format = "channels_first"))
#         cnn.add(BatchNormalization())
#         ########new add#######
#         cnn.add(Flatten())
#         # cnn.add(Dense(512, activation='relu'))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Dense(512, activation='relu'))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Dense(256, activation='relu'))
#         # cnn.add(BatchNormalization())
#         # cnn.add(Dense(256, activation='relu'))
#         # cnn.add(BatchNormalization())
#         # # cnn = Dropout(0.25)(cnn)
#         cnn.add(Dense(128, activation='relu'))
#         cnn.add(BatchNormalization())
#         cnn.add(Dense(64, activation='relu'))
#         cnn.add(BatchNormalization())
#         cnn.add(Dropout(0.25))
#         cnn.add(Dense(1, activity_regularizer=regularizers.l2(0.01)))
#         cnn.compile(optimizer=self.adam, metrics=[r_squared, "mean_squared_error"], loss="mean_absolute_error")
#         # print(cnn.summary())
#         self.model = cnn
#         print(cnn.summary())
#
#         es = EarlyStopping(min_delta=0.0001, patience=200, mode='min', monitor='val_loss', restore_best_weights=True,
#                            verbose=1)
#         rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=100,verbose = 1,min_lr = 0.0005)
#
#         if len(xVal) == 0 and len(yVal) == 0:
#             #use GPU
#             gpu_no = 0
#             with tf.device("/gpu:" + str(gpu_no)):
#                 print("this is to run gpu")
#                 cnn.fit(xtrain,
#                         ytrain,
#                         batch_size=self.batch_size,
#                         epochs=self.epochs,
#                         validation_split=0.3,
#                         callbacks=[es, rp],
#                         # callbacks = [es],
#                         verbose = 1)
#         else:
#             gpu_no = 0
#             with tf.device("/gpu:" + str(gpu_no)):
#                 print("this is to run gpu")
#                 cnn.fit(xtrain,
#                            ytrain,
#                            batch_size=self.batch_size,
#                            epochs=self.epochs,
#                            validation_data=(xVal, yVal),
#                             callbacks=[es, rp],
#                            # callbacks=[es],
#                             verbose = 2)
#         self.model = cnn
#
#
#     def fitting_single_column(self, xtrain, ytrain, xVal=[], yVal=[]):
#         """
#         :param xtrain:(group_numbers, img_height,img_weight,group_size)
#         :param ytrain:shape(-1,1)
#         :param xVal:validation dataset
#         :param yVal:validation dataset
#         :return:save model into self, no returns
#         """
#         # define CNN model
#         model_inputs = Input(shape=self.input_shape)
#         cnn_model = Conv2D(32, kernel_size=4, activation='relu', padding='same')(model_inputs)
#         cnn_model = Conv2D(32, kernel_size=4, activation='relu', padding='same')(cnn_model)
#         cnn_model = MaxPooling2D(pool_size=4)(cnn_model)  # strides: If None, it will default to pool_size.
#         cnn_model = Conv2D(64, kernel_size=4, activation='relu', padding='same')(cnn_model)
#         cnn_model = Conv2D(64, kernel_size=4, activation='relu', padding='same')(cnn_model)
#         cnn_model = MaxPooling2D(pool_size=4)(cnn_model)
#         cnn_model = Flatten()(cnn_model)
#         output = Dense(256, activation='relu')(cnn_model)
#         output = BatchNormalization()(output)
#         output = Dense(128, activation='relu')(output)
#         output = BatchNormalization()(output)
#         output = Dense(64, activation='relu')(output)
#         output = BatchNormalization()(output)
#         output = Dropout(0.25)(output)
#         output = Dense(1, activity_regularizer=regularizers.l2(0.01),name = "output")(output)
#
#         model = Model(inputs=model_inputs, outputs=output)
#         print(model.summary())
#
#
#         model.compile(optimizer=self.adam, metrics=[r_squared, "mean_squared_error"], loss={"output": "mean_absolute_error"})
#
#         es = EarlyStopping(min_delta=0.001, patience=12, mode='min', monitor='val_loss', restore_best_weights=True,
#                            verbose=1)
#         rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20,verbose = 2,min_lr = 0.0005)
#
#         if len(xVal) == 0 and len(yVal) == 0:
#             model.fit(xtrain,
#                            ytrain,
#                            batch_size=self.batch_size,
#                            epochs=self.epochs,
#                            validation_split=0.3,
#                            callbacks=[es, rp])
#         else:
#             model.fit(xtrain,
#                            ytrain,
#                            batch_size=self.batch_size,
#                            epochs=self.epochs,
#                            validation_data=(xVal, yVal),
#                            callbacks=[es, rp])
#         self.model = model
#
#     def fitting_individual_model(self, xtrain, ytrain, xVal=[], yVal=[]):
#         """
#
#         work terrible; because each time it's updated due to one column's loss, then learn another column label value and updates again in that way;
#         therefore, it cannot fit all columns.
#         :param xtrain:(group_numbers, img_height,img_weight,group_size)
#         :param ytrain:shape(-1,6)
#         :param xVal:validation dataset
#         :param yVal:validation dataset
#         :return:save model into self, no returns
#         """
#         # define CNN model
#         model_inputs = Input(shape=self.input_shape)
#         cnn_model = Conv2D(32, kernel_size=4, activation='relu', padding='same')(model_inputs)
#         cnn_model = Conv2D(32, kernel_size=4, activation='relu', padding='same')(cnn_model)
#         cnn_model = MaxPooling2D(pool_size=4)(cnn_model)  # strides: If None, it will default to pool_size.
#         cnn_model = Conv2D(64, kernel_size=4, activation='relu', padding='same')(cnn_model)
#         cnn_model = Conv2D(64, kernel_size=4, activation='relu', padding='same')(cnn_model)
#         cnn_model = MaxPooling2D(pool_size=4)(cnn_model)
#         cnn_model = Flatten()(cnn_model)
#         cnn_model = Dense(256, activation='relu')(cnn_model)
#         cnn_model = BatchNormalization()(cnn_model)
#         # cnn_model.add(Dropout(0.25))
#         cnn_model = Dense(128, activation='relu')(cnn_model)
#         cnn_model = BatchNormalization()(cnn_model)
#         cnn_model = Dense(64, activation='relu')(cnn_model)
#         cnn_model = BatchNormalization()(cnn_model)
#         cnn_model = Dropout(0.25)(cnn_model)
#
#         # multiple outputs: 'rotX', 'rotY', 'rotZ', 'transX', 'transY', 'transZ'
#         rotX = Dense(1, activity_regularizer=regularizers.l2(0.01),name = "rotX")(cnn_model)
#         rotY = Dense(1, activity_regularizer=regularizers.l2(0.01), name = "rotY")(cnn_model)
#         rotZ = Dense(1, activity_regularizer=regularizers.l2(0.01),name = "rotZ")(cnn_model)
#         transX = Dense(1, activity_regularizer=regularizers.l2(0.01),name = "transX")(cnn_model)
#         transY = Dense(1, activity_regularizer=regularizers.l2(0.01),name = "transY")(cnn_model)
#         transZ = Dense(1, activity_regularizer=regularizers.l2(0.01),name = "transZ")(cnn_model)
#
#         model = Model(inputs=model_inputs, outputs=[rotX, rotY, rotZ, transX, transY, transZ])
#         print(model.summary())
#
#         model.compile(optimizer=self.adam, metrics=[r_squared, "mean_squared_error"], loss={"rotX": "mean_absolute_error",
#                                                                                          "rotY": "mean_absolute_error",
#                                                                                          "rotZ": "mean_absolute_error",
#                                                                                          "transX": "mean_absolute_error",
#                                                                                          "transY": "mean_absolute_error",
#                                                                                          "transZ": "mean_absolute_error"})
#         self.model = model
#
#         es = EarlyStopping(min_delta=0.001, patience=100, mode='min', monitor='val_loss', restore_best_weights=True,
#                            verbose=1)
#         rp = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20,verbose = 2,min_lr = 0.0005)
#
#         rotX, rotY, rotZ = ytrain[:, 0], ytrain[:, 1], ytrain[:, 2],
#         transX, transY, transZ = ytrain[:, 3], ytrain[:, 4], ytrain[:, 5]
#
#         if len(xVal) == 0 and len(yVal) == 0:
#                 self.model.fit(xtrain,
#                                [rotX, rotY, rotZ, transX, transY, transZ],
#                                batch_size=self.batch_size,
#                                epochs=self.epochs,
#                                validation_split=0.3,
#                                callbacks=[es, rp])
#         else:
#             self.model.fit(xtrain,
#                            [rotX, rotY, rotZ, transX, transY, transZ],
#                            batch_size=self.batch_size,
#                            epochs=self.epochs,
#                            validation_data=(xVal, yVal),
#                            callbacks=[es, rp])
#
#     def evaluate(self, x_test, y_test):
#         """
#         loss is MSE
#         monitor: r Squared; MAE
#         """
#         loss,score_r_square,mae = self.model.evaluate(x_test, y_test)
#         return loss,score_r_square,mae
#
#     def predict(self, xtest):
#         y_pred = self.model.predict(xtest)
#         return y_pred
#
#     def get_model(self):
#         return self.model
#
#     def save_model(self,name = "name"):
#         self.model.save('{}.h5'.format(name))
#
# class cnn_lstm():
#     """
#     input image size:(img_row,img_col,1)
#     default times = 3
#     default epochs =1000
#     default batch_size = 156
#     """
#     def __init__(self, batch_size=156, epochs=1000,input_shape = None):
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.input_shape = input_shape
#             # define CNN model
#         print('\nThis is self-defined cnn model\n')
#         model = Sequential()
#         model.add(TimeDistributed(Conv2D(32, kernel_size=4, activation='relu', padding='same'),input_shape=self.input_shape))  # input is a greyscale pic
#         model.add(TimeDistributed(Conv2D(32, kernel_size=4, activation='relu', padding='same')))
#         model.add(TimeDistributed(MaxPooling2D(pool_size=4)))  # strides: If None, it will default to pool_size.
#         model.add(TimeDistributed(Conv2D(64, kernel_size=4, activation='relu', padding='same')))
#         model.add(TimeDistributed(Conv2D(64, kernel_size=4, activation='relu', padding='same')))
#         model.add(TimeDistributed(MaxPooling2D(pool_size=4)))
#         model.add(TimeDistributed(Flatten()))
#         model.add(TimeDistributed(Dense(64,activation='relu')))
#         model.add(TimeDistributed(Dense(32,activation='relu')))
#         model.add(TimeDistributed(Dropout(0.25)))
#         model.add(LSTM(units=256,return_sequences=True,name="lstm_layer1"))
#         model.add(TimeDistributed(Dense(64,activation = 'relu')))
#         model.add(TimeDistributed(Dropout(0.25)))
#         model.add(TimeDistributed(Dense(1,kernel_regularizer = regularizers.l2(0.001))))
#         model.compile(optimizer='adam', metrics=[r_squared,'mean_absolute_error'], loss='mean_squared_error')
#         print(model.summary())
#         self.model = model
#         pass
#
#
#     def fitting(self,xtrain, ytrain,xVal= [], yVal = []):
#         """
#         :param xtrain:
#         :param ytrain:
#         :param xVal:
#         :param yVal:
#         :return:
#         """
#         es = EarlyStopping(min_delta=0.001, patience=8, mode='min', monitor='val_loss',restore_best_weights = True,verbose = 2)
#         rp = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4,verbose = 2)
#
#         if len(xVal) ==0 and len(yVal) == 0:
#             self.model.fit(xtrain, ytrain,
#                     batch_size=self.batch_size,
#                     epochs=self.epochs,
#                     validation_split = 0.3,
#                     callbacks=[es, rp])
#         else:
#             self.model.fit(xtrain, ytrain,
#                     batch_size=self.batch_size,
#                     epochs=self.epochs,
#                     validation_data=(xVal, yVal),
#                     callbacks=[es, rp])
#
#     def evaluate(self, x_test, y_test):
#         """
#         loss is MSE
#         monitor: r Squared; MAE
#         """
#         loss,score_r_square,mae= self.model.evaluate(x_test, y_test)
#         return loss,score_r_square,mae
#
#     def predict(self, xtest):
#         y_predict = self.model.predict(xtest)
#         return y_predict
#
#     def get_model(self):
#         return self.model
#
#     def save_model(self):
#         self.model.save('cnn_lstm.h5')
