import os
import argparse

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--path', default='E:\data_new\\', type=str,
                                       help='')
    args = parse.parse_args()
    path = args.path
#path = 'C:\\Users\张飞\Desktop\图片暂存'
# 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)

    n = 0
    for i in fileList:
    # 设置旧文件名（就是路径+文件名）
        oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

    # 设置新文件名
        newname = path + os.sep + fileList[n] + '.apk'

        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)

        n += 1
