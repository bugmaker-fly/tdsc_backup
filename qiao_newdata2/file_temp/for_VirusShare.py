'''
以model0来计算多指标
'''

import csv
import numpy as np
import joblib as jl
from skmultilearn.dataset import load_from_arff
from skmultilearn.ext import Meka
from embedding import to_arff_for_label

def get_label_count(y_pred):
    count_list = np.array([0,0,0,0,0,0])
    for i in range(y_pred.shape[0]):
        for j in range(6):
            if y_pred[i][j] == 1:
                count_list[j] += 1

    return count_list

def merge_x_y(path1, temp_train_data_y):
    '''
    将x与x的预测结果y合并为一个csv
    '''
    temp_y = np.array(temp_train_data_y.todense())

    f = open(path1, 'r', encoding='utf-8')
    path_out = 'D:\lab_related\my_code\qiao_newdata\data\\train_temp_xy.csv'
    f_w = open(path_out, 'w')
    writer = csv.writer(f_w, lineterminator='\n')
    flag = True #标记第一行
    n = 0
    for line in f.readlines():
        list_feature = line.strip().split(',')
        list_feature2 = list_feature[:-6]
        if flag:
            flag = False
            writer.writerow(list_feature)
        else:
            for i in range(temp_y.shape[1]):
                list_feature2.append(temp_y[n][i])
            n += 1
            writer.writerow(list_feature2)
    return path_out


path1 = 'C:\Java\jdk-12.0.2\\bin\java.exe'
meka_classpath = r'D:\lab_related\meka-release-1.9.2\lib\\'#MEKA
meka_classifiers = ['RAkELd']#RT后面的是LP
weka_classifiers = ['trees.RandomForest']

for meka_c in meka_classifiers:
    meka_classifier = 'meka.classifiers.multilabel.' + meka_c
    for weka_c in weka_classifiers:
        weka_classifier = 'weka.classifiers.' + weka_c
        meka = Meka(
            meka_classifier=meka_classifier,  # Binary Relevance: classifier chains
            weka_classifier=weka_classifier,  # with Naive Bayes single-label classifier
            meka_classpath=meka_classpath,  # obtained via download_meka
            java_command=path1  # path to java executable
        )

        #x_test, y_test = load_from_arff('D:\lab_related\my_code\qiao_newdata\data\\test\\manual_test.arff', label_count=6)
        x_train, y_train = load_from_arff('D:\lab_related\my_code\qiao_newdata\data\\train\\manual_train.arff', label_count=6)
        model0 = meka.fit(X=x_train, y=y_train)
        print(model0.get_params())
        base_path = 'D:\lab_related\my_code\qiao_newdata\data'


        arff_path = to_arff_for_label(1, 1)
        # 预测出y
        temp_train_data_x, _ = load_from_arff(arff_path, label_count=6)
        temp_train_data_y = model0.predict(temp_train_data_x)
        # 将y与x合并在一起
        temp_data_xy = merge_x_y(base_path + '\\train_temp.csv', temp_train_data_y)
        y_narry = np.array(temp_train_data_y.todense())
        re = get_label_count(y_pred=y_narry)
        print(re)