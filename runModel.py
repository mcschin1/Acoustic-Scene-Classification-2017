# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:16:06 2019

@author: sonat
"""
import pandas as pd
from sys_folder import tfsetting
import os
import shutil
import numpy as np
from dataProcessing.DataProcessing import *
import model as m
import pickle
from sys_folder.learningRateScheduler import PolynomialDecay
from datetime import datetime
from sys_folder import util
from keras import applications

from keras import backend as K
       
def getModel(modelName,dataConfig,learning_rate=1e-2):
    model = getattr(m,modelName)(dataConfig,learning_rate)
    return model

def main(modelName):
    modelName='resNet2017'
    max_epochs=100
    batch_size=32
    #norm= Normalizer(filename=normMap)
    dataConfig = DataConfig(256,256,1,15)
    f,original = tfsetting.startLogging('out/{0}_out.log'.format(modelName))
    for i in range(1,5):
        K.clear_session()
        i=1
        fold = '{0}_fold'.format(modelName)+str(i)
        PREDICTION_FOLDER = "predictions_"+fold
        DATA_DIR = "../Processed/melscale82_256/"
        best_weight='../weight/Best_'+fold+'.h5'
        final_weight='../weight/final_'+fold+'.h5'
        historyPath = '../history/history_'+fold+'.pickle'
    
        trainset = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold{0}_train.csv'.format(str(i)),
                                   index_col='filename')
        testset = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold{0}_evaluate.csv'.format(str(i)),
                                   index_col='filename')
        
        if os.path.exists('logs/' + PREDICTION_FOLDER):
            shutil.rmtree('logs/' + PREDICTION_FOLDER)
        model = getModel(modelName,dataConfig)
        model.summary()
        checkpoint = tfsetting.ModelCheckpoint(best_weight, monitor='val_loss', verbose=0, save_best_only=True)
        #early = tfsetting.EarlyStopping(monitor="val_loss", mode="min", patience=15)
        tb = tfsetting.TensorBoard(log_dir='.\\logs\\{0}'.format(PREDICTION_FOLDER), write_graph=True)
        #lrScheduler = tfsetting.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=1, verbose=1)
        schedule = PolynomialDecay(maxEpochs=max_epochs, initAlpha=1e-2, power=5)
        callbacks_list = [ checkpoint, tb,tfsetting.LearningRateScheduler(schedule)]
        train_generator = MixUpGenerator(dataConfig,DATA_DIR,trainset.index,
                                           trainset.label_idx,batch_size=batch_size)
        val_generator = DataGenerator(dataConfig,DATA_DIR,testset.index,
                                         testset.label_idx,batch_size=batch_size)
        currenttime = datetime.now().strftime("%H:%M:%S")
        print("=================Current Start time for fold {0} is {1}".format(i,currenttime))
        history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                             epochs=max_epochs,steps_per_epoch=trainset.shape[0] // batch_size, 
                                             use_multiprocessing=False,workers=1,max_queue_size=1)
        model.save_weights(final_weight)
        print("=================End time for fold {0} is {1}".format(i,currenttime))
        # Store data (serialize)
        with open(historyPath, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tfsetting.closeLogging(f,original)

modelList=[
           'WRN'
    ]
for i in modelList:
    print("===================Start model {0} =================================".format(i))
    main(i)
    print("============Program Completed Successfully===============")

def mainRaw():
    K.clear_session()
    modelName='rawResWCNN'
    i=1
    DATA_DIR = "../Processed/2019Task1a/raw/"
    PREDICTION_FOLDER = "predictions_"+modelName
    f,original = tfsetting.startLogging('out/2019_norm/{0}_out.log'.format(modelName))
    max_epochs=100
    '''
    trainset = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold{0}_train.csv'.format('1'),
                               index_col='filename')
    testset = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold{0}_evaluate.csv'.format('1'),
                               index_col='filename')
    '''
    trainset = pd.read_csv('../Dataset/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold{0}_train_df.csv'.format(str(i)),
                               index_col='filename')
    testset = pd.read_csv('../Dataset/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold{0}_evaluate_df.csv'.format(str(i)),
                               index_col='filename')
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)
    dataConfig = dp.DataConfig(0,32000,1,10,is_1D=True)
    model = getModel(modelName,dataConfig)
    model.summary()
    checkpoint = tfsetting.ModelCheckpoint('Best_rawResNet.h5', monitor='val_loss', verbose=2, save_best_only=True)
    #early = tfsetting.EarlyStopping(monitor="val_loss", mode="min", patience=15)
    tb = tfsetting.TensorBoard(log_dir='.\\logs\\{0}'.format(PREDICTION_FOLDER)+'\\fold_1', write_graph=True)
    #lrScheduler = tfsetting.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=1, verbose=1)
    schedule = PolynomialDecay(maxEpochs=max_epochs, initAlpha=1e-2, power=5)
    callbacks_list = [ checkpoint, tb,tfsetting.LearningRateScheduler(schedule)]
    train_generator = dp.DataGenerator(dataConfig,DATA_DIR,trainset.index,
                                       trainset.label_idx,batch_size=32)
    val_generator = dp.DataGenerator(dataConfig,DATA_DIR,testset.index,
                                     testset.label_idx,batch_size=32)
    history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                         epochs=max_epochs, use_multiprocessing=False, workers=2, max_queue_size=5)
    model.save_weights('Final_rawResNet.h5')
    # Store data (serialize)
    with open('Final_rawResNet_history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tfsetting.closeLogging(f,original)
 
def main2019(modelName):
    #modelName='CBAM'
    foldername='2019_82_256'
    max_epochs=100
    #dataConfig = dp.DataConfig(256,470,1,10)
    dataConfig = DataConfig(256,256,1,10)
    K.clear_session()
    model = getModel(modelName,dataConfig)
    #modelName=modelName+'_noMix'
    i=1
    fold = '{0}_fold'.format(modelName)+str(i)
    PREDICTION_FOLDER = "predictions_"+fold
    DATA_DIR = "../Processed/2019Task1a/melscare82_256/"
    best_weight='../weight/{0}/Best_'.format(foldername)+fold+'.h5'
    final_weight='../weight/{0}/final_'.format(foldername)+fold+'.h5'
    historyPath = '../history/{0}/history_'.format(foldername)+fold+'.pickle'
    
    trainset = pd.read_csv('../Dataset/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold{0}_train_df.csv'.format(str(i)),
                               index_col='filename')
    testset = pd.read_csv('../Dataset/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold{0}_evaluate_df.csv'.format(str(i)),
                               index_col='filename')
    
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)
    f,original = tfsetting.startLogging('out/{0}/{1}_out.log'.format(foldername,modelName))
    model.summary()
    checkpoint = tfsetting.ModelCheckpoint(best_weight, monitor='val_loss', verbose=0, save_best_only=True)
    #early = tfsetting.EarlyStopping(monitor="val_loss", mode="min", patience=15)
    tb = tfsetting.TensorBoard(log_dir='.\\logs\\2019_norm\\{0}'.format(PREDICTION_FOLDER), write_graph=True)
    lrScheduler = tfsetting.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, cooldown=1, verbose=1)
    schedule = PolynomialDecay(maxEpochs=max_epochs, initAlpha=1e-2, power=5)
    callbacks_list = [ checkpoint, tb,tfsetting.LearningRateScheduler(schedule),lrScheduler]
    ''' 
    train_generator = MixUpGenerator(dataConfig,DATA_DIR,trainset.index,
                                       trainset.label_idx,batch_size=32,alpha=0.4) #or alpha=0.7
    '''
    train_generator = DataGenerator(dataConfig,DATA_DIR,trainset.index,
                                     trainset.label_idx,batch_size=32)

    val_generator = DataGenerator(dataConfig,DATA_DIR,testset.index,
                                     testset.label_idx,batch_size=32,shuffle=False)
    currenttime = datetime.now().strftime("%H:%M:%S")
    print("=================Current Start time for fold {0} is {1}".format(i,currenttime))
    history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                         epochs=max_epochs, use_multiprocessing=False
                                         ,workers=1,max_queue_size=1)
    model.save_weights(final_weight)
    print("=================End time for fold {0} is {1}".format(i,currenttime))
    # Store data (serialize)
    with open(historyPath, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tfsetting.closeLogging(f,original)

'''
           'SENET','res_SENET','mobileNetV2','denseNet','Xception',
            'WRN','Inception','highwayNet','resNET','resNET_V2','resnXet','VGG2','AlexNet2','firstCNN'
 
'''
modelList=[
          'resnXet','mobileNet','resNET3'
          ,'resNet2017','shuffleNet','CBAM'
    ]
for i in modelList:
    print("===================Start model {0} =================================".format(i))
    main2019(i)
    print("============Program Completed Successfully===============")