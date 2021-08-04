# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_file', required=True, help='输出矢量数据')
    parser.add_argument('--input_file', required=True, help='输入矢量')
    parser.add_argument('--layer_name', help='矢量图层名称')
    parser.add_argument('--field', required=True, default='DN', help='字段名称')

    args = parser.parse_args()

    input_file = args.input_file
    layer_name = args.layer_name
    if layer_name is None:
        layer_name = os.path.splitext(os.path.basename(input_file))[0]

    # grass location
    location = '/grassdb/test_grass'
    # init grass
    os.system(f"grass --text -c -e {args.input_file} {location}")
    # exec dissolve shell
    os.system(f"grass {location}/PERMANENT --exec sh dissolve.sh "
              f"{args.output_file} {input_file} {layer_name} {args.field}")