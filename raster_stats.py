# -*- coding: utf-8 -*-

try:
    import gdal
except ModuleNotFoundError:
    from osgeo import gdal

import os
import json

jf = 'd:/tmp/0815.json'

# jl = json.loads(jf,encoding='utf-8')
with open(jf) as jh:
    jl = json.loads(jh.read())

# with open("d:/tmp/0815.json", 'w') as jh:
#     json.dump(jl, jh)

c = 0
for j in jl:
    pan_file = j['pan_file']
    mss_file = j['mss_file']
    if not os.path.exists(os.path.dirname(pan_file)):
        os.makedirs(os.path.dirname(pan_file))
    if not os.path.exists(os.path.dirname(mss_file)):
        os.makedirs(os.path.dirname(mss_file))
    with open(pan_file, 'w') as p:
        p.write('a')
    with open(mss_file, 'w') as m:
        m.write('b')
    # first_6_pan = os.path.basename(pan_file)[:6]
    # first_6_mss = os.path.basename(mss_file)[:6]
    # if first_6_pan != first_6_mss:
    #     c += 1
    #     print(c)
    #     print(os.path.basename(j['pan_file']), os.path.basename(j['mss_file']))
    # if first_6.startswith('GF2'):
    #     if j['gsdx_mss'] == 2.88e-5:
    #         print(os.path.basename(j['pan_file']))
    # print(os.path.basename(j['pan_file'])[:10], j['gsdx_pan'], j['gsdx_mss'])