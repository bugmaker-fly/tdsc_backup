'''
前期数据的处理，按batuch读取数据并和原来的数据集合并，返回arff格式
'''
import pandas as pd
import csv

pn = 'qiao_backup6'
label_num = 6
feature_num = 531
title='@relation \'unlabled: -C 6\''
f_feature = '/home/rtfeng/fei/'+pn+'/data/final_feature_500.txt'

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


def read_batch(batch,batch_size):
    f_train = open('/home/rtfeng/fei/'+pn+'/data/train_temp.csv', 'w', encoding='utf-8')  #暂定训练集csv存放位置
    writer = csv.writer(f_train,lineterminator='\n')


    f_4000 = open('/home/rtfeng/fei/'+pn+'/data/VirusShare_5000.csv', 'r', encoding='utf-8')
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

def to_arff_for_label(batch,batch_size):
    '''
    用于给unlabel训练集打标签
    '''
    path = read_batch(batch,batch_size)
    #out_arff_path = 'D:\lab_related\my_code\qijing_qiao\data\\train\\train_temp.arff'  #只有batch_size条
    arff_path = csv2arff(path, f_feature, feature_num+label_num)
    return arff_path

def to_arff_for_acc(train_path, out_path):
    return
