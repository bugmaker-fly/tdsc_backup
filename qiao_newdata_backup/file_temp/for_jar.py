import glob
import os
import shutil
import zipfile
import argparse
from functools import partial
from multiprocessing import Pool

from subprocess import PIPE, Popen

a = 0
def test_apk(file):
    #print(file)
    process = Popen(['java', '-jar', '../main.jar', file], stdout=PIPE, stderr=PIPE)
    process.communicate()
    #print('-------------------------------------')
    #f_txt = open(file+'_api.txt', 'r')
    #for line in f_txt.readlines():
    #    oldline = line
    #    newline = line.replace(',','\n').replace(';','\n')
        #f_txt = f_txt.replace(',','\n').replace(';','\n')
    #f_re = open(file+'_api.txt', 'w')
    #f_re.write(newline)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', default='E:\data_new\\', type=str,
                           help='path for run')
    parse.add_argument('--process_num', default=1, type=int, help='number of process')
    args = parse.parse_args()

    files = glob.glob(args.path + '/*.apk')
    #print(type(files))
    pool = Pool(processes=args.process_num)
    pool.map(test_apk, files)
    #resultList = pool.apply(partial_job, files)
    pool.close()
    pool.join()
    #print(a)
