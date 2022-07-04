import os
#base_path = '/home/rtfeng/fei/qiao_newdata2/result2_b'
#file_path = base_path + '/CDT_LMT_8/' + 'best_train_set_now_4.csv'
file_path = '/home/rtfeng/fei/qiao_newdata_backup/result2_b/CDN_J48_8/best_train_set_now_4.csv'
f = open(file_path, 'r', encoding='utf-8')
for line in f.readlines():
    line_list = line.strip().split(',')
    flag = True
    for i in range(6):
        index =  -1 * i - 1
        if line_list[index] != '0':
            flag = False

    if flag:
        print(line_list[0])
