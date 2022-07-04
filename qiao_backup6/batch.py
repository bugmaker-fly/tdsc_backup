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
from embedding import to_arff_for_label
from embedding import csv2arff

path1 = '/home/rtfeng/fei/jdk-17.0.1/bin/java'
meka_classpath = r'/home/rtfeng/fei/meka-release-1.9.2/lib/'#MEKA
meka_classifiers = ['PS']
weka_classifiers = ['trees.RandomForest']

pn = 'qiao_backup6'
label_num = 6
feature_num = 531
title='@relation \'unlabled: -C 6\''
f_feature = '/home/rtfeng/fei/'+pn+'/data/final_feature_500.txt'

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')


def get_acc2(y_pred, y_true, args):
    right_count = 0
    y_pred = np.array(y_pred.todense())
    y_true = np.array(y_true.todense())
    for i in range(y_true.shape[0]):
        flag = True
        flag2 = False
        for j in range(args.label_count):   
            if y_true[i][j]==0 and y_pred[i][j]==1:
                flag = False
                break
        for j in range(args.label_count):
            if y_pred[i][j] == 1:
                flag2 = True
                break
        if flag and flag2:
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
    path_out = '/home/rtfeng/fei/'+pn+'/data/train_temp_xy.csv'
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
    return_path = '/home/rtfeng/fei/'+pn+'/data/temp_train_data.csv'
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


def merge_csv2(path1, path2):
    f_path1 = open(path1, 'r', encoding='utf-8')
    f_path2 = open(path2, 'r', encoding='utf-8')
    return_path = '/home/rtfeng/fei/'+pn+'/data/temp_train_data_shuffle.csv'
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

def read_batch2(batch,batch_size):
    f_train = open('/home/rtfeng/fei/'+pn+'/data/train_temp.csv', 'w', encoding='utf-8')  #csv
    writer = csv.writer(f_train,lineterminator='\n')

    f_4000 = open('/home/rtfeng/fei/'+pn+'/data/del_input_newdata_shuffle.csv', 'r', encoding='utf-8')
    #reader = csv.reader(f_4000)
    line0 = f_4000.readline()
    list_line0 = line0.strip().split(',')
    list_line0.append('MESSAGE')
    list_line0.append('blackmail')
    list_line0.append('KEEP')
    list_line0.append('phone')
    list_line0.append('ad')
    list_line0.append('internet')
    #print(list_line0)
    writer.writerow(list_line0)

    #f_train.write('\n')
    i = 0
    for line in f_4000.readlines():
        if i >= batch*batch_size and i < batch*batch_size+batch_size:
            list_line = line.strip().split(',')
            for j in range(6):
                list_line.append('0')
                #print(list_line)
            writer.writerow(list_line)
        i += 1
    path = '/home/rtfeng/fei/'+pn+'/data/train_temp.csv'
    return path

def to_arff_for_label2(batch,batch_size):
    path = read_batch2(batch,batch_size)
    #out_arff_path = 'D:\lab_related\my_code\qijing_qiao\data\\train\\train_temp.arff'  #batch_size
    arff_path = csv2arff(path, f_feature, feature_num+label_num)
    return arff_path

def get_len(temp):
    f = open(temp,'r')
    re = 0
    lines = f.readlines()
    re = len(lines)
    re = re - 150
    return re
def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', default=500, type=int,
                       help='迭代次数')
    parse.add_argument('--label_count', default=6, type=int,
                       help='label的维度')
    parse.add_argument('--random_seed', default=200, type=int,
                       help='随机种子')
    #parse.add_argument('--model_path', default='E:\lab_related\mycode\qijing_qiao385\model', type=str,
     #                  help='存放训练模型的路径')

    args = parse.parse_args()

    #全局变量
    base_path = '/home/rtfeng/fei/'+pn+'/data'
    f_feature = base_path + '/final_feature_500.txt'
    label_num = 6
    feature_num = 531
    title = '@relation \'unlabled: -C 6\''
    f_batch = open('/home/rtfeng/fei/qiao_newdata2/batch/PS_RandomForest_4/batch.csv', 'w')
    f_batch.write('batch,acc,data')
    f_batch.write('\n')
    g_batch = 0
    g_acc = 0
    g_data = 0
    best_acc = 0
    for meka_c in meka_classifiers:
        for weka_c in weka_classifiers:
            meka_classifier = 'meka.classifiers.multilabel.' + meka_c
            weka_classifier = 'weka.classifiers.' + weka_c
            meka = Meka(
                meka_classifier=meka_classifier,  # Binary Relevance: classifier chains
                weka_classifier=weka_classifier,  # with Naive Bayes single-label classifier
                meka_classpath=meka_classpath,  # obtained via download_meka
                java_command=path1  # path to java executable
            )

            def get_model(train_path, test_path, model_path_f, i):
                '''
                根据手动标记数据集得到acc作为初始acc用于后续添加数据对比
                '''
                # 训练模型
                x_train, y_train = load_from_arff(train_path, label_count=label_num)
                x_test, y_test = load_from_arff(test_path, label_count=label_num)
                model = meka.fit(X=x_train, y=y_train)

                # 保存模型
                model_path = model_path_f + '/model' + str(i) + '.pkl'
                jl.dump(model, model_path)

                # 对于初始model备份
                if i == 0:
                    model_path = model_path_f + '/model' + '1' + '.pkl'
                    jl.dump(model, model_path)

                # 计算acc
                y_pred = model.predict(x_test)
                # print(y_pred.shape())
                acc = get_acc2(y_pred, y_test, args)
                label_acc_list_temp = get_label_acc(y_pred=y_pred, y_true=y_test)
                return acc, model_path,label_acc_list_temp
            # >
            for batch_size in [4]:
                break_flag = False
                train_path = base_path + '/train/manual_train.arff'
                test_path = base_path + '/test/manual_test.arff'
                dir_r = meka_c + '_' + weka_c.strip().split('.')[1] + '_' + str(batch_size)
                print(dir_r)
                save_path = '/home/rtfeng/fei/'+pn+'/result/' + dir_r #存放结果
                mkdir(save_path)
                model_path = save_path + '/model'
                mkdir(model_path)
                f_detail = open(save_path+'/detail.csv', 'w')
                acc0, model0, label_acc_list = get_model(train_path, test_path, model_path, 0)
                f_detail.write('begin_acc2')
                f_detail.write(',')
                f_detail.write(str(acc0))
                f_detail.write('\n')
                f_detail.write('begin_label_acc')
                f_detail.write(',')
                f_batch.write('0,')
                f_batch.write(str(acc0))
                f_batch.write(',0')
                f_batch.write('\n')
                for i, l_acc in enumerate(label_acc_list):
                    f_detail.write(str(l_acc))
                    if i !=5:
                        f_detail.write(',')
                f_detail.write('\n')
                print(f'acc 0: {acc0}')
                best_model = model0
                best_acc = acc0
                best_train_set_csv = base_path + '/best_train_set.csv' #训练集
                copy(best_train_set_csv, base_path + '/manual_train.csv')
                for batch in range(5450//batch_size):
                    g_batch += 1
                    #batch_size条数据的arff格式
                    arff_path = to_arff_for_label(batch,batch_size)
                    #预测出y
                    temp_train_data_x,_ = load_from_arff(arff_path, label_count=args.label_count)
                    model = jl.load(best_model)
                    temp_train_data_y = model.predict(temp_train_data_x)
                    #将y与x合并在一起
                    temp_data_xy = merge_x_y(base_path+'/train_temp.csv', temp_train_data_y)
                    #合成训练集的暂定csv版本
                    temp_train_data = merge_csv(best_train_set_csv, temp_data_xy, args)
                    shuffle(temp_train_data, args)
                    #得到temp_train_data.arff
                    temp_train_data_arff = csv2arff(temp_train_data, f_feature, feature_num+label_num)

                    acc_now, model_now, label_acc_list_now = get_model(temp_train_data_arff, test_path, model_path, batch+1)
                    print(f'acc {batch+1}: {acc_now}')
                    if acc_now > best_acc:
                        g_data = get_len(temp_train_data)
                        f_batch.write(str(g_batch))
                        f_batch.write(',')
                        f_batch.write(str(acc_now))
                        f_batch.write(',')
                        f_batch.write(str(g_data))
                        f_batch.write('\n')
                        best_model = model_now
                        best_acc = acc_now
                        copy(best_train_set_csv, temp_train_data)
                        f_detail.write('batch')
                        f_detail.write(',')
                        f_detail.write(str(batch))
                        f_detail.write('\n')
                        f_detail.write('acc2')
                        f_detail.write(',')
                        f_detail.write(str(acc_now))
                        f_detail.write('\n')
                        f_detail.write('label_acc')
                        f_detail.write(',')
                        for j, l_acc_now in enumerate(label_acc_list_now):
                            f_detail.write(str(l_acc_now))
                            if j != 5:
                                f_detail.write(',')
                        f_detail.write('\n')

                    #删除没必要的模型文件
                    model_file = glob.glob(model_path + '/*.pkl')
                    for file in model_file:
                        if file != best_model and file != (model_path+'/model0.pkl'):
                            os.remove(file)

                best_csv = save_path + '/best_train_set_now.csv'
                copy(best_csv, best_train_set_csv)
            
            #enter >=
            acc_begin = best_acc
            print(f'acc begin: {acc_begin}')
            model_folder = 'PS_RandomForest_4'
            result_base = '/home/rtfeng/fei/'+pn+'/result'
            model_path = result_base + '/' + model_folder + '/model'
            model_file = glob.glob(model_path + '/*.pkl')
            m_len = len(model_file)
            print(model_file)
            if m_len > 1:
                model_begin = model_file[1]
            else:
                model_begin = model_file[0]
            if model_file[0] != model_path + '/model0.pkl':
                model_begin = model_file[0]
            if True:
                del_path = result_base + '/' + model_folder + '/best_train_set_now.csv' 
                del_out_path = base_path +  '/del_input_newdata_shuffle.csv'
                length = del_input_4000('/home/rtfeng/fei/'+pn+'/data/input_derbin_new.csv', del_path, del_out_path)
            
            test_path = base_path + '/test/manual_test.arff'
            save_path = '/home/rtfeng/fei/'+pn+'/result2/' + model_folder  #  -----------------------------------------------------
            mkdir(save_path)
            model_path = save_path + '/model_1'
            mkdir(model_path)
            f_detail = open(save_path + '/detail.csv', 'w')
            source_detali = open(result_base + '/' + model_folder + '/detail.csv', 'r', encoding='utf-8')
                                                                                                #>
            for source_line in source_detali.readlines():
                for j, str1 in enumerate(source_line.strip().split(',')):
                    f_detail.write(str1) #-------------------------------- _0 ---------------------------------
                    if j != len(source_line.strip().split(','))-1:
                        f_detail.write(',')
                f_detail.write('\n')
            best_model = model_begin
            jl.dump(jl.load(best_model), save_path + '/model_1' + '/model0.pkl')
            best_acc = acc_begin
            best_train_set_csv = base_path + '/best_train_set.csv'  # 
            copy(best_train_set_csv, del_path)
            print(f'acc 0: {acc_begin}')
            
            for epoch in range(4):
                if epoch != 0:
                    model_path = save_path + '/model_' + str(epoch+1)
                    mkdir(model_path)
                    length = del_input_4000('/home/rtfeng/fei/'+pn+'/data/del_input_newdata_shuffle.csv', save_path+'/best_train_set_now_' + str(epoch)+'.csv', base_path+'/del_input_newdata_shuffle_backup.csv')
                    copy('/home/rtfeng/fei/'+pn+'/data/del_input_newdata_shuffle.csv', base_path+'/del_input_newdata_shuffle_backup.csv')
                    shuffle(base_path+'/del_input_newdata_shuffle.csv', args)
                for batch in range((length-1) // batch_size):
                    g_batch += 1
                    arff_path = to_arff_for_label2(batch, batch_size)
                                                # y
                    temp_train_data_x, _ = load_from_arff(arff_path, label_count=args.label_count)
                    model = jl.load(best_model)
                    temp_train_data_y = model.predict(temp_train_data_x)
                                                                                                                # yx
                    temp_data_xy = merge_x_y(base_path + '/train_temp.csv', temp_train_data_y)
                                                                                                                                                # csv
                    temp_train_data, flag_add = merge_csv2(best_train_set_csv, temp_data_xy)
                    shuffle(temp_train_data, args)
                # temp_train_data.arff
                    acc_now = 0.0
                    if flag_add:
                        temp_train_data_arff = csv2arff(temp_train_data, f_feature, feature_num + label_num)
                        acc_now, model_now, label_acc_list_now = get_model(temp_train_data_arff, test_path, model_path,batch + 1)
                        print(f'epoch 0 acc {batch + 1}: {acc_now}')
                    if float(acc_now) >= float(best_acc):
                        g_data = get_len(temp_train_data)
                        f_batch.write(str(g_batch))
                        f_batch.write(',')
                        f_batch.write(str(acc_now))
                        f_batch.write(',')
                        f_batch.write(str(g_data))
                        f_batch.write('\n')
                        best_model = model_now
                        best_acc = acc_now
                        copy(best_train_set_csv, temp_train_data)
                        f_detail.write('acc2_'+str(epoch+1))
                        f_detail.write(',')
                        f_detail.write(str(acc_now))
                        f_detail.write('\n')
                        f_detail.write('label_acc')
                        f_detail.write(',')
                        for j,l_acc_now in enumerate(label_acc_list_now):
                            f_detail.write(str(l_acc_now))
                            if j != 5:
                                f_detail.write(',')
                        f_detail.write('\n')
                    model_file = glob.glob(model_path + '/*.pkl')
                    for file in model_file:
                        if file != best_model and file != (model_path + '/model0.pkl'):
                            os.remove(file)
                best_csv = save_path + '/best_train_set_now_' + str(epoch+1) + '.csv'
                copy(best_csv, best_train_set_csv)



if __name__ == "__main__":
    main()
