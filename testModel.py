# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:05:41 2020

@author: kekxi
"""
import pandas as pd
from sys_folder import tfsetting
import os
import shutil
import numpy as np
from dataProcessing import DataProcessing as dp
import model as m
import pickle
from sys_folder.learningRateScheduler import PolynomialDecay
from keras import backend as K
from sys_folder import util

def getAccuracyResult(resultDF):
    count = 0
    for index,row in resultDF.iterrows():
        if row['Label'] == row['Target'][0]:
            count+=1
    print(count/len(resultDF.index))
    return count/len(resultDF.index)

def getAllResult(dataset,labelList):
    fullResultList = []
    for i,label in enumerate(labelList):
        resultList = dataset.loc[dataset['Label']==label]
        fullResultList.append(float(getAccuracyResult(resultList)))
    return fullResultList

def getModel(modelName,dataConfig,learning_rate=1e-2):
    model = getattr(m,modelName)(dataConfig,learning_rate)
    return model

def test():
    K.clear_session()
    i=1
    modelName='resNet2017'
    PREDICTION_FOLDER = "predictions_"+modelName
    DATA_DIR = "../Processed/melscale82_256/"
    best_weight = '../weight/best_{0}_fold{1}.h5'.format(modelName,i)
    final_weight = '../weight/final_{0}_fold{1}.h5'.format(modelName,i)
    #historyPath = 'history_'+modelName+'.pickle'
    testSet = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold{0}_test.csv'.format('1')
                          ,index_col=0)
    testSet['filename'] = testSet.filename.apply(lambda x: x.split('.')[0])
    testSet.set_index('filename',inplace=True)
    label_idx = util.loadLabel()
    testSet['act_label_idx']=testSet.Label.apply(lambda x: label_idx[x])
    dataConfig = DataConfig(256,256,1,15)
    model = getModel(modelName,dataConfig)
    model.load_weights(final_weight)
    train_generator = dp.DataGenerator(dataConfig,DATA_DIR,testSet.index,testSet.act_label_idx,
                                       batch_size=32,shuffle=False)
    predictions = model.predict_generator(train_generator,use_multiprocessing=False,
                                             workers=1, max_queue_size=1,verbose=1)
    
    score = model.evaluate_generator(train_generator,use_multiprocessing=False,
                                             workers=1, max_queue_size=1,verbose=1)
    np.save('predictions_out_'+modelName+'.npy',predictions,allow_pickle=True)
    
    top_1 = np.array(list(label_idx.keys()))[np.argsort(-predictions,axis=1)[:,:1]]
    testSet["Target"]=top_1.tolist()
    fullResultDF = pd.DataFrame({'label':list(label_idx)})
    fullResultDF['score'] = getAllResult(testSet,label_idx)
    fullResultDF['score'].mean()
    

def evaluation(modelName):
    K.clear_session()
    
    modelName='resNet2017'
    dataConfig = dp.DataConfig(256,256,1,15)
    model = getModel(modelName,dataConfig)
    DATA_DIR="../Processed/melscale82_256/"
    evalSet = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-evaluation/evaluation_setup/evaluate.csv'
                       ,index_col=['filename'])
    predtotal = None
    label_idx = util.loadLabel()
    for i in range(1,5):
        i=1
        prediction_folder='../result/predictions/out_{0}_fold{1}.npy'.format(modelName,i)
        best_weight = '../weight/best_{0}_fold{1}.h5'.format(modelName,i)
        final_weight = '../weight/final_{0}_fold{1}.h5'.format(modelName,i)
        model.load_weights(best_weight)
        train_generator = dp.DataGenerator(dataConfig,DATA_DIR,evalSet.index,evalSet.act_label_idx,
                                       batch_size=32,shuffle=False)
        predictions = model.predict_generator(train_generator,use_multiprocessing=False,
                                             workers=1, max_queue_size=1,verbose=1)
        score = model.evaluate_generator(train_generator,use_multiprocessing=False,
                                             workers=1, max_queue_size=1,verbose=1)
        np.save(prediction_folder,predictions,allow_pickle=True)
        if predtotal is None:
            predtotal = predictions
        else:
            predtotal += predictions
    predtotal = predtotal/4
    np.save('../result/predictions/out_{0}_avg.npy'.format(modelName),predtotal,allow_pickle=True)
    #top_1 = np.array(list(label_idx.keys()))[np.argsort(-predictions,axis=1)[:,:1]]
    top_avg = np.array(list(label_idx.keys()))[np.argsort(-predtotal,axis=1)[:,:1]]
    evalSet["Target"]=top_avg.tolist()
    fullResultDF = pd.DataFrame({'label':list(label_idx)})
    fullResultDF['score'] = getAllResult(evalSet,label_idx)
    fullResultDF['score'].mean()
    print('===============Classification score for {0} : {1}'.format(modelName,str(fullResultDF.mean())))
    evalSet.to_csv("../result/metaData/metaData_{0}.csv".format(modelName))
    fullResultDF.to_csv("../result/classification_score/score_{0}.csv".format(modelName))

'''
'SENET','res_SENET','mobileNet','mobileNetV2','denseNet','Xception',
            'WRN','Inception','highwayNet','resNET','resNET_V2','resnXet','VGG','AlexNet','firstCNN'
'''
modelList =['VGG','AlexNet','firstCNN']
f,original = tfsetting.startLogging('../result/out_best.log')
for i in modelList:
    print("===================Start model {0} =================================".format(i))
    evaluation(i)
    print("============Program Completed Successfully===============")
tfsetting.closeLogging(f,original)    

'''
==============
Preprocessing steps
================
    evalSet = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold{0}_test.txt'.format('1'),
                           header=None,names=['filename'])
    fullList = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-development/meta.txt',
                           header=None,names=['filename'])
    evalSet['filename'] = evalSet.filename.apply(lambda x: x.split('/')[1])
    fullList['Label'] =  fullList.filename.apply(lambda x: x.split('\t')[1])
    fullList['filename'] = fullList.filename.apply(lambda x: x.split('\t')[0].split('/')[1])
    finalSet = pd.merge(evalSet,fullList,on='filename')
    finalSet.to_csv("../Dataset/TUT-acoustic-scenes-2017-development/evaluation_setup/fold1_test.csv")

evalSet1 = pd.read_csv('../Dataset/TUT-acoustic-scenes-2017-evaluation/evaluation_setup/evaluate.csv'
                       ,index_col=['filename'])

evalSet['label'] = evalSet.filename.apply( lambda x : x.split('\t')[1])
evalSet['filename'] = evalSet.filename.apply( lambda x : x.split('\t')[0])
evalSet['filename'] = evalSet.filename.apply( lambda x : x.split('.')[0])
evalSet['filename'] = evalSet.filename.apply( lambda x : 'melscale'+x)
evalSet.set_index('filename',inplace=True)
label_idx = util.loadLabel()
evalSet['act_label_idx']=evalSet.label.apply(lambda x: label_idx[x])
evalSet.rename(columns={'label':'Label'},inplace=True)
evalSet.to_csv("../Dataset/TUT-acoustic-scenes-2017-evaluation/evaluation_setup/evaluate.csv")
'''
