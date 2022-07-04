import os
import argparse
import numpy as np
import csv
import joblib as jl
import random
import pandas as pd
from skmultilearn.dataset import load_from_arff
from skmultilearn.ext import Meka

path1 = 'E:\APP\java\\bin\java.exe'
meka_classpath = r'C:\Users\326\scikit_ml_learn_data\meka\meka-release-1.9.2\lib\\'#MEKA
meka = Meka(
    meka_classifier="meka.classifiers.multilabel.CDN",  # Binary Relevance: classifier chains
    weka_classifier="weka.classifiers.trees.LMT",  # with Naive Bayes single-label classifier
    meka_classpath=meka_classpath,  # obtained via download_meka
    java_command=path1  # path to java executable
)


label_num = 6
feature_num = 531
title = '@relation \'unlabled: -C 6\''

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')

def get_acc1(y_pred, y_true, args):
    right_count = 0
    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_true.todense())
    for i in range(y_true.shape[0]):
        flag = True
        for j in range(args.label_count):
            if y_true[i][j] != y_pred[i][j]:
                flag = False
        if flag:
            right_count += 1
    return format(right_count / y_true.shape[0], '.8f')

def get_acc2(y_pred, y_true, args):
    right_count = 0
    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_true.todense())
    for i in range(y_true.shape[0]):
        flag = True
        for j in range(args.label_count):
            if y_true[i][j]==0 and y_pred[i][j]==1:
                flag = False
        if flag:
            right_count += 1
    return format(right_count / y_true.shape[0], '.8f')

def get_label_acc(y_pred, y_true):
    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_true.todense())
    right_count_list = np.array([0,0,0,0,0,0])
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if y_true[i][j] == y_pred[i][j]:
                right_count_list[j] += 1
    label_acc = right_count_list / y_true.shape[0]
    return label_acc

def merge_x_y(path1, temp_train_data_y):
    '''
    将x与x的预测结果y合并为一个csv
    '''
    temp_y = np.array(temp_train_data_y.todense())

    f = open(path1, 'r', encoding='utf-8')
    path_out = 'E:\lab_related\mycode\qijing_qiao385\data\\train_temp_xy.csv'
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

def merge_csv(path1, path2, args):
    '''
    合并两个csv文件,path1是原数据集，path2是预测的
    '''
    f_path1 = open(path1, 'r', encoding='utf-8')
    f_path2 = open(path2, 'r', encoding='utf-8')
    return_path = 'E:\lab_related\mycode\qijing_qiao385\data\\temp_train_data.csv'
    f_out_path = open(return_path, 'w')
    writer = csv.writer(f_out_path, lineterminator='\n')
    flag = True
    for line1 in f_path1.readlines():
        line_list1 = line1.strip().split(',')
        writer.writerow(line_list1)
    _ = f_path2.readline()
    list = []
    for line2 in f_path2.readlines():
        line_list2 = line2.strip().split(',')
        list.append(line_list2)
    writer.writerows(list)
    f_out_path.close()

    return return_path

def merge_csv2(path1, path2):
    '''
    将path2中path1不存在的部分和path1合并起来
    '''
    f_path1 = open(path1, 'r', encoding='utf-8')
    f_path2 = open(path2, 'r', encoding='utf-8')
    return_path = 'E:\lab_related\mycode\qijing_qiao385\data\\temp_train_data_shuffle.csv'
    f_out_path = open(return_path, 'w')
    writer = csv.writer(f_out_path, lineterminator='\n')
    flag = False
    id = []
    for line1 in f_path1.readlines():
        line_list1 = line1.strip().split(',')
        id.append(line_list1[0])
        writer.writerow(line_list1)
    _ = f_path2.readline()
    list = []
    for line2 in f_path2.readlines():
        line_list2 = line2.strip().split(',')
        if line_list2[0] not in id:
            list.append(line_list2)
            flag = True
    writer.writerows(list)
    f_out_path.close()

    return return_path, flag

def copy(path1, path2):
    '''
    把path2的内容copy到path1作为其内容
    '''
    f_path1 = open(path1, 'w',)
    writer = csv.writer(f_path1,lineterminator='\n')
    f_path2 = open(path2, 'r')
    for line in f_path2.readlines():
        line_list = line.strip().split(',')
        writer.writerow(line_list)

def shuffle(csv_path, args):
    '''
    打乱csv中数据的顺序
    '''
    seed = args.random_seed
    random.seed(seed)
    f = open(csv_path, 'r', encoding='utf-8')
    first_line = f.readline()
    lines = f.readlines()
    random.shuffle(lines)
    f.close()
    f2 = open(csv_path, 'w')
    writer = csv.writer(f2, lineterminator='\n')
    first_line_list = first_line.strip().split(',')
    writer.writerow(first_line_list)
    for line in lines:
        line_list = line.strip().split(',')
        writer.writerow(line_list)

def del_input_4000(source_path, del_path, outpath):
    source_f = open(source_path, 'r', encoding='utf-8')
    outpath_f = open(outpath, 'w')
    writer = csv.writer(outpath_f, lineterminator='\n')
    del_f = open(del_path, 'r', encoding='utf-8')
    del_list = []
    len = 0
    flag = False
    for del_line in del_f.readlines():
        if flag:
            del_l = del_line.strip().split(',')
            del_list.append(del_l[0])
        else:
            flag = True
    for source_line in source_f.readlines():
        source_l = source_line.strip().split(',')
        if source_l[0] not in del_list:
            writer.writerow(source_l)
            len += 1
    return len

def csv2arff(fpath,f_feature,n):
    list_feature = []  # 这个地方每次都需要改
    df=pd.read_csv(fpath, engine='python')

    datatype=[]
    for i in range(0,label_num+feature_num):
        datatype.append('numeric')

    #open feature-txt
    path_feature = f_feature
    feature_file = open(path_feature, "r")

    for line in feature_file.readlines():
        ss = line.strip()
        list_feature.append(ss)
    feature_file.close()

    #need to be changed each time
    list_feature.append('MESSAGE')
    list_feature.append('blackmail')
    list_feature.append('KEEP')
    list_feature.append('phone')
    list_feature.append('ad')
    list_feature.append('internet')

    fpath=fpath[:fpath.find('.csv')]+'.arff'
    f = open(fpath,'w')
    f.write(title+'\n')
    f.write('\n')
    for i in range(0,feature_num+label_num):
        f.write('@attribute {} {}\n'.format(list_feature[i],datatype[i]))
    f.write('\n')
    f.write('@data\n')
    # df.sgape[0]返回行数，  df.shape[1]返回列数
    f.close()
    f = open(fpath, 'a')
    for i in range(df.shape[0]):
        #print(df.shape[0])

        item=df.iloc[i,:df.shape[1]]
        num=0
        flag = True
        for j in item:
            num += 1
            if flag:
                flag = False
            else:
                j_str = str(j)
                if num <= n:
                    f.write(j_str + ',')
                else:
                    f.write(j_str)
                    f.write('\n')


    f.close()
    return fpath

def get_model(train_path, test_path1, test_path2, args):
    '''
    根据手动标记数据集得到acc作为初始acc用于后续添加数据对比
    '''
    #训练模型
    x_train, y_train = load_from_arff(train_path, label_count=args.label_count)
    x_test1, y_test1 = load_from_arff(test_path1, label_count=args.label_count)
    x_test2, y_test2 = load_from_arff(test_path2, label_count=args.label_count)
    model = meka.fit(X=x_train, y=y_train)

    #保存模型
    model_path = args.model_path + '\model'  + '.pkl'
    jl.dump(model, model_path)

    #计算acc
    y_pred1 = model.predict(x_test1)
    y_pred2 = model.predict(x_test2)
    acc1 = get_acc2(y_pred1, y_test1, args)
    acc2 = get_acc2(y_pred2, y_test2, args)
    return acc1, acc2, model_path


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', default=500, type=int,
                       help='迭代次数')
    parse.add_argument('--label_count', default=6, type=int,
                       help='label的维度')
    parse.add_argument('--random_seed', default=200, type=int,
                       help='随机种子')
    parse.add_argument('--model_path', default='E:\lab_related\mycode\qijing_qiao385\\test_stability_result\CDN_LMT_8', type=str,
                       help='存放训练模型的路径')

    args = parse.parse_args()

    result_path = 'E:\lab_related\mycode\qijing_qiao385\\test_stability_result'
    f_feature = result_path + '\\final_feature_500.txt'
    test_path1 = result_path + '\\train_and_test.arff'
    test_path2 = result_path + '\manual_test.arff'
    train_temp_csv = result_path + '\CDN_LMT_8\\best_train_set_now5.csv'
    del_csv = result_path + '\\train_and_test.csv'
    train_csv = result_path + '\CDN_LMT_8\\train_csv.csv'
    len = del_input_4000(train_temp_csv, del_csv, train_csv)

    train_arff = csv2arff(train_csv, f_feature, feature_num+label_num)
    acc1, acc2, model = get_model(train_arff, test_path1,test_path2, args)
    print(f'acc1:{acc1}')
    print(f'acc2:{acc2}')


if __name__ == "__main__":
    main()

