# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:45:54 2022

@author: DELL
"""
import os
import shutil
import glob
from PIL import Image
import subprocess


def move_file(file, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    shutil.move(file, directory)


def label(in_dir, label_txt):
    """Open and label images in a directory"""
    move_dir = os.path.join(in_dir, 'labeled')
    label_range = ['1', '2', '3', '4', '5']
    for pic in glob.glob(os.path.join(in_dir, '*.jpg')):
        fp = open(pic, 'rb')
        img = Image.open(fp)
        img.show()
        qua = input("Quality label:")
        print(qua)
        if qua == 'd':
            fp.close()
            os.remove(pic)
        if qua in label_range:
            with open(label_txt, 'a') as out:
                out.write("{0}\t{1}\n".format(os.path.basename(pic), qua))
            fp.close()
            move_file(pic, move_dir)
        img.close()


if __name__ == '__main__':
    in_dir = r"E:\S2\缩略图"
    label_txt = 'd:/label.txt'
    # label(in_dir, label_txt)
    
    img_dir = r'E:\S2\label'
    os.chdir(img_dir)
    label_txt = 'label.txt'
    with open(label_txt) as fh:
        line = fh.readline()
        while(line):
            # print(line)
            img, label = line.rstrip('\n').split('\t')
            if not os.path.isfile(img):
                print(img)
                line = fh.readline()
                continue
            move_file(img, label)
            line = fh.readline()
            # break
    
    
    
        
        
        
        