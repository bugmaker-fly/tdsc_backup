f_batch = open('C:\\Users\张飞\Desktop\\file_temp\CT_RandomForest_32_batch.csv', 'r', encoding='utf-8')
batch = []
acc = []
data = []
t = 0
acc_now = '0'
data_now = '0'
_ = f_batch.readline()
for line in f_batch.readlines():
    line_list = line.strip().split(',')
    batch.append(line_list[0])
    acc.append(line_list[1])
    data.append(line_list[2])
f_re = open('D:\lab_related\my_code\qijing_qiao\graph_CT_RandomForest_32.csv', 'w', encoding='utf-8')
f_re.write('batch,acc,data\n')
for i in range(170*5):
    if str(i) in batch:
        acc_now = acc[t]
        data_now = data[t]
        t += 1
    f_re.write(str(i))
    f_re.write(',')
    f_re.write(acc_now)
    f_re.write(',')
    f_re.write(data_now)
    f_re.write('\n')