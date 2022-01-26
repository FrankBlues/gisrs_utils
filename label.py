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

if __name__ == '__main__':
    in_dir = r"E:\S2\缩略图"
    label_txt = 'd:/label.txt'
    for pic in glob.glob(os.path.join(in_dir, '*.jpg')):
        fp = open(pic, 'rb')
        img = Image.open(fp)
        img.show()
        qua = input("Quality label:")
        print(qua)
        if qua == 'd':
            fp.close()
            os.remove(pic)
        if qua in ['1', '2', '3', '4', '5']:
            with open(label_txt, 'a') as out:
                out.write("{0}\t{1}\n".format(os.path.basename(pic), qua))
            fp.close()
            shutil.move(pic, os.path.join(in_dir, 'labeled'))
        img.close()
    
        
        
        
        