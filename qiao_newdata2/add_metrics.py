import os
import argparse
import numpy as np
import csv
import joblib as jl
import glob
import pandas as pd
import random
from skmultilearn.dataset import load_from_arff
from skmultilearn.ext import Meka
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error

def get_acc2(y_pred, y_true):
    right_count = 0
    for i in range(y_true.shape[0]):
        flag = True
        flag2 = False
        for j in range(y_true.shape[1]):
            if y_true[i][j]==0 and y_pred[i][j]==1:
                flag = False
                break
        for j in range(y_true.shape[1]):
            if y_pred[i][j] == 1:
                flag2 = True
                break
        if flag and flag2:
            right_count += 1
    return format(right_count / y_true.shape[0], '.8f')

base_path = '/home/rtfeng/fei/qiao_newdata2/data'
model_paths = [#'/home/rtfeng/fei/qiao_newdata/result2/CC_RandomForest_4/model_2/model300.pkl',
       # '/home/rtfeng/fei/qiao_newdata_backup/result2/CC_RandomForest_32/model_1/model168.pkl',
        #'/home/rtfeng/fei/qiao_backup/result2/CDN_REPTree_32/model_2/model97.pkl',
        #'/home/rtfeng/fei/qiao_backup/result2/CT_RandomForest_32/model_4/model1.pkl',
        #'/home/rtfeng/fei/qiao_backup2/result2/CC_RandomForest_16/model_1/model54.pkl',
        #'/home/rtfeng/fei/qiao_backup3/result2/PS_RandomForest_32/model_1/model169.pkl',
        #'/home/rtfeng/fei/qiao_backup3/result2/CT_RandomForest_16/model_1/model58.pkl',
        #'/home/rtfeng/fei/qiao_backup4/result2/PS_RandomForest_16/model_2/model3.pkl',
        #'/home/rtfeng/fei/qiao_backup5/result2/CDN_J48_4/model_1/model614.pkl',
        #'/home/rtfeng/fei/qiao_backup5/result2/CDN_J48_16/model_1/model0.pkl',
        #'/home/rtfeng/fei/qiao_backup5/result/CDN_J48_32/model/model0.pkl',
        #'/home/rtfeng/fei/qiao_backup5/result2/CT_RandomForest_4/model_2/model1254.pkl',
        #'/home/rtfeng/fei/qiao_backup6/result2/PS_RandomForest_4/model_2/model12.pkl',
        #'/home/rtfeng/fei/qiao_backup7/result2/CDN_REPTree_16/model_4/model3.pkl',
        #'/home/rtfeng/fei/qiao_backup8/resul2/CDN_REPTree_4/model_1/model846.pkl'
        '/home/rtfeng/fei/for_qijing_paper/model/DREBIN/Detection-Training/CDN_LMT.pkl'
        ]
f_re = open('/home/rtfeng/fei/qiao_newdata2/add_metrics_test.csv', 'w')
f_re.write('mwka,sample-acc,haming-loss,one-loss,instance-f1\n')
for model_path in model_paths:
    #/home/rtfeng/fei/qiao_newdata/CDN_LMT_8/model/model100.pkl
    id = model_path.strip().strip('/').split('/')[-3]
    f_re.write(str(id))
    f_re.write(',')

    best_model = jl.load(model_path)
    test_path = base_path + '/test/manual_test.arff'
    x_test, y_test = load_from_arff(test_path, label_count=6)
    y_pred = best_model.predict(x_test)

    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_test.todense())
    # 测试sample-acc
    acc2 = get_acc2(y_pred, y_true)
    hammingloss = hamming_loss(y_pred=y_pred, y_true=y_true)
    oneloss = zero_one_loss(y_pred=y_pred, y_true=y_true)
    instance_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    f_re.write(str(acc2))
    f_re.write(',')
    f_re.write(str(hammingloss))
    f_re.write(',')
    f_re.write(str(oneloss))
    f_re.write(',')
    f_re.write(str(instance_f1))
    f_re.write('\n')
