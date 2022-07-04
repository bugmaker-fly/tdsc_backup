import glob
import os
import shutil
import zipfile
import argparse
from functools import partial
from multiprocessing import Pool

def test_apk(file):
    name_list = []
    if zipfile.is_zipfile(file):  # 这里初步判断是否是压缩包
        try:
            zfile = zipfile.ZipFile(file)
            name_list = zfile.namelist()
            zfile.close()
        except:
            if os.path.exists:
                os.remove(file)
            print('ZipFile error!')
            return
    if 'classes.dex' in name_list and 'AndroidManifest.xml' in name_list:
        print(file)
    else:
        #print('delete')
        if os.path.exists:
            os.remove(file)
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', default='E:\data_new\\', type=str,
                           help='待处理文件夹的绝对路径')
    parse.add_argument('--process_num', default=5, type=int, help='设置的并行进程数')
    args = parse.parse_args()

    files = glob.glob(args.path + '*')
    # print('-----------------files-------------')
    # print(files)

    pool = Pool(processes=args.process_num)
    partial_job = partial(test_apk)
    resultList = pool.map(partial_job, files)
    pool.close()
    pool.join()

