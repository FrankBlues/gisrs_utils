# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:03:21 2022

@author: DELL
"""
import os
from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    
    filename = 'd:/piccc.jpg'
    text = '20211128'
    # 创建绘画对象
    image= Image.open(filename)
    draw= ImageDraw.Draw(image)
    width, height= image.size # 宽度，高度
    size= int(0.05*width)  # 字体大小(可以调整0.04)
    myfont= ImageFont.truetype(r'C:\Windows\Fonts\CASTELAR.TTF', size=size) # 80, 4032*3024  # MISTRAL.TTF Inkfree.ttf
    #fillcolor= '#000000' # RGB黑色   # CASTELAR.TTF
    fillcolor= (235, 80, 10)

    d_width, d_height=0.7*width, 0.94*height # 字体的相对位置
    draw.text((d_width, d_height), text, font=myfont, fill=fillcolor) 

    new_filename = 'd:/picc-2.png'
    image.save(new_filename)