#!/usr/bin/python
# coding:utf-8

# -----------------------------------------------------------------------------------------------
# @Author     : qiang.wang
# @Date       : 2020-10-22 16:27
# @File       : rsiutils.py
# @Description: Some common methods commonly used
# -----------------------------------------------------------------------------------------------

import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

def check_dir(dir):
    """Determine whether the directory exists, if it does not exist, create the directory"""
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except FileExistsError:
            print("WARNING: Directory already exists")

def check_parent_dir(filename):
    """Determine whether the directory where the file exists exists, if it does not exist, create the directory"""
    path = os.path.dirname(filename)
    if not os.path.exists(path) and path.strip():
        try:
            os.makedirs(path)
        except FileExistsError:
            print("WARNING: Directory already exists")

def check_file_exist(filename):
    """Determine whether the result file exists or not, exit the program"""
    if not os.path.isfile(filename):
        print("ERROR: The result file does not exist : %s" % filename)
        exit(1)

def check_dir_exist(result_dir):
    """Determine whether the result directory exists or not, exit the program"""
    if not os.path.isdir(result_dir):
        print("ERROR: The result directory does not exist : %s" % result_dir)
        exit(1)

def check_empty_dir(directory):
    """Determine whether the directory empty or not, exit the program if empty"""
    if len(os.listdir(directory)) == 0:
        print("ERROR: The directory is empty : %s" % directory)
        exit(1)

def write_file(filename, string):
    """Write string to file"""
    check_parent_dir(filename)
    f = open(filename, 'w', encoding='utf-8')
    f.write(string)
    f.close()

def write_json_file(filename, content_dict):
    """Write dict to file"""
    check_parent_dir(filename)
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(content_dict, f)

def write_list_file(filename, file_list, sep='\n'):
    """Write list to file.

    Args:
        filename(str):写出的文件;
        file_list(list):文件列表;
        sep(str):分隔符,默认换行'\n'.

    """
    if os.path.exists(filename):
        os.remove(filename)
    check_parent_dir(filename)
    f = open(filename, 'a', encoding='utf-8')
    for i, line in enumerate(file_list):
        # 最后一项后面不加分隔符
        if i == len(file_list) - 1:
            f.write(line)
        else:
            f.write(line + sep)
    f.close()

def read_file(filename):
    """Print the file content"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    print("The xml file is : %s, print the file :\n%s" % (filename, content))

def read_json_file(filename):
    """read json content"""
    with open(filename, "r", encoding='utf-8') as f:
        content_dict = json.load(f)
    # print("The json file is : %s, print the file :\n%s" % (filename, content_dict))
    return content_dict

def get_file_content(filename):
    """Print the file content"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def run(cmd):
    try:
        res = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as err:
        print('ERROR:', err)
        exit(1)
    else:
        # print(res.args)
        print('Return status code: ', res.returncode)

def run_without_exit(cmd):
    try:
        res = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as err:
        print('WARNING:', err)
        #exit(1)
    else:
        # print(res.args)
        print('Return status code: ', res.returncode)

def get_image_file(path,
                   extensions=['tif', 'img', 'tiff', 'TIF', 'IMG', 'TIFF'],
                   image_type='all'):
    """List all files within a directory with the given extensions.
    
    Args:
        path (str): File path.
        extensions (list): File suffix list.
        image_type (str): Image type in ['all', 'mss', 'pan', 'dsm'], default 'all'.

    """
    img_files = []
    for ext in extensions:
        file_found = Path(path).glob('**/*.{}'.format(ext))
        for f in file_found:
            f_str = str(f)
            fn = os.path.basename(f_str)
            if f_str not in img_files:
                if fn.lower().startswith('bj3') or fn.lower().startswith('triplesat'):
                    if fn.endswith('browser.tif'):
                        continue
                if image_type == 'all':  # all images found.
                    img_files.append(f_str)
                elif image_type == 'mss':  # multi-spectral images.
                    if any([k in fn for k in ['favm_', '_mux_', 'MUX', '_MS_', 'MSS', 'M2AS']]):
                        img_files.append(f_str)
                elif image_type == 'pan':  # multi-spectral images.
                    if fn.startswith('GF7'):
                        if '_BWD_'in fn:
                            img_files.append(f_str)
                    elif any([k in fn for k in ['navp_', '_nad_', 'NAD', 'PAN', '_P_', 'pan', 'P2AS']]):
                        img_files.append(f_str)
                elif image_type == 'dsm':  # dsm input images.
                    if fn.startswith('SV'):
                        if fn.endswith("PAN1.tiff") or fn.endswith("PAN2.tiff"):
                            img_files.append(f_str)
                    elif any([k in fn for k in ['navp_', '_nad_', 'NAD', 'favp_', '_fwd_', '-FWD', '_FWD_', '_bwd_', '-BWD', '_BWD_', 'P2AS']]):
                        img_files.append(f_str)
                else:
                    raise ValueError("image_type should be in ['all', 'mss', 'pan', 'dsm'].")
    return img_files

def change_one_xml(xmlfile):
    """"""
    doc = ET.parse(xmlfile)
    root = doc.getroot()
    sub1 = root.find('OnlyRGB')
    sub1.text = 'true'
    doc.write(xmlfile)
    print('----------done--------')


def get_image_type(file_name):
    """Judge the image type according to file name, return types include [pan, mss, None]"""
    if file_name.endswith('browser.tif'):
        return
    # 高景数据
    if file_name.startswith('SV'):
        _pan, _mss = ['P1', 'P2'], ['M1', 'M2']
        for i in range(len(_pan)):
            if file_name.endswith(_pan[i] + '.img'):
                return 'pan'
            elif file_name.endswith(_mss[i] + '.img'):
                return 'mss'
    elif file_name.endswith('TIL'):
        if 'P2AS' in file_name:  # world view
            return 'pan'
        elif 'M2AS' in file_name:  # world view
            return 'mss'
    elif file_name.upper().startswith('GF7'):
        if 'BWD' in file_name:
            return 'pan'
        elif 'MUX' in file_name:
            return 'mss'
    else:
        if any([key_word in file_name for key_word in ['navp_', '_nad_', 'NAD', 'PAN', 'pan', 'DOMP', '_P_']]):
            return 'pan'
        elif any([key_word in file_name for key_word in ['favm_', '_mux_', 'MUX', 'MSS', 'DOMM', '_MS_', '_MS1_']]):
            return 'mss'


if __name__ == '__main__':
    # change_one_xml('../xml/DsmBOSBundle.xml')
    # read_file('../xml/DsmBOSBundle.xml')
    # change_one_xml('../xml/XQFusionCmd.xml')
    # read_file('../xml/XQFusionCmd.xml')
    # content = read_file('../config/XQFusionCmd.xml')
    # write_file("XQFusionCmd.xml", "hello")
    # print(get_image_file(r'D:\work\data\影像样例', image_type='pan'))
    # write_list_file('d:/filelist1.txt', ['/tmp/1.tif', '/tmp/2.tif', '/tmp/3.tif'], sep='\n')
    c = read_json_file(r'D:\tmp\image_parallel.json')
    print(c[0])
