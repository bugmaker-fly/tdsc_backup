'''
change to \n, and move to anather dir
'''
import os
import argparse
import shutil
import glob


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', default='/home/rtfeng/fei/VirusShare_malware/', type=str,
                           help='source path')
    args = parse.parse_args()
    dir_n = args.path.strip('/').split('/')[-1]   # eg: VirusShare_00400
    txt_files = glob.glob(args.path+'/*.txt')

    final_path = '/home/rtfeng/fei/VirusShare_malware/txt/'+dir_n + '/'
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    for txt_file in txt_files:
        id = txt_file.strip('/').split('/')[-1]  # eg: a.txt
        for line in open(txt_file, 'r', encoding='utf-8').readlines():
            newline = line.replace(',', '\n').replace(';', '\n')
        f_changed = open(final_path+id, 'w')
        f_changed.write(newline)
        os.remove(txt_file)
