# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:45:29 2018

用于调用fmask命令行工具计算哨兵2数据云掩膜,见pythonfmask网站：
http://pythonfmask.org/en/latest/

注意:
    需要先将哨兵2数据组织成为亚马逊云哨兵2存放类似方式，见：
    https://registry.opendata.aws/sentinel-2/

Todo:
    * 更通用一些，不必首先将数据按照亚马逊云哨兵2数据组织方式

"""

from __future__ import print_function, division
import os
import glob
from subprocess import run, PIPE

from SAFE_process import unzip_band_meta_image


def rearange_s2_data(safe_dir, root_dir, wildcard):
    """重新组织哨兵2数据,将哨兵2原数据(SAFE)解压并组织成类似亚马逊云哨兵数据存放格式.

    Args:
        safe_dir (str) : Directory storing the sentinel 2 SAFE data.
        root_dir (str) : Directory where the data unzipped to.
        wildcard (str) : Condition used by the glob tool to list file needed.

    """

    for z in glob.glob(os.path.join(safedir, wildcard)):
        unzip_band_meta_image(z, rootdir)


if __name__ == '__main__':

    safedir = 'I:/sentinel'
    rootdir = 'I:/sentinel/unzip'
    rearange_s2_data(safedir, rootdir, 'S2*20181104*.zip')

    # 绝对路径
    pyenv = 'C:/ProgramData/Anaconda3/envs/py333/'
    pyscript = pyenv + 'Scripts/fmask_'
    pythonEXE = pyenv + 'python.exe'
    fmask_sentinel2Angles_script = pyscript + 'sentinel2makeAnglesImage.py'
    fmask_sentinel2Stacked_script = pyscript + 'sentinel2Stacked.py'

    for root, dirs, files in os.walk(rootdir):
        for onedir in dirs:
            d = os.path.join(root, onedir)
            print(d)
            os.chdir(d)
            outVRT = 'allbands.vrt'
            outAngle = 'angles.img'
            outCloudImg = 'cloud.img'
            mincloudsize = 200
            cloudbufferdistance = 80
            shadowbufferdistance = 40
            cloudprobthreshold = 0.225
            nirsnowthreshold = 0.11
            greensnowthreshold = 0.1

            print('转换gdal vrt数据格式')
            r = run('gdalbuildvrt -resolution user -tr 20 20 -separate {} \
                    B01.jp2 B02.jp2 B03.jp2 B04.jp2 B05.jp2 B06.jp2 B07.jp2 \
                    B08.jp2 B8A.jp2 B09.jp2 B10.jp2 B11.jp2 B12.jp2'
                    .format(outVRT), shell=True, stderr=PIPE)

            assert(r.stderr == b'' and r.returncode == 0)

            print('生成角度数据')

            r = run('{0} {1} -i ./MTD_TL.xml -o {2}'.format(pythonEXE,
                    fmask_sentinel2Angles_script, outAngle),
                    shell=True, stderr=PIPE)

            assert(r.stderr == b'' and r.returncode == 0)

            print('计算云掩膜')
            r = run('{0} {1} -a {vrtfile} -z {angleImg} -o {cloudImg} \
                    --mincloudsize {mincloudsize} \
                    --cloudbufferdistance {cloudbufferdistance} \
                    --shadowbufferdistance {shadowbufferdistance} \
                    --cloudprobthreshold {cloudprobthreshold} \
                    --nirsnowthreshold {nirsnowthreshold} \
                    --greensnowthreshold {greensnowthreshold}'
                    .format(pythonEXE, fmask_sentinel2Stacked_script,
                            vrtfile=outVRT, angleImg=outAngle,
                            cloudImg=outCloudImg, mincloudsize=mincloudsize,
                            cloudbufferdistance=cloudbufferdistance,
                            shadowbufferdistance=shadowbufferdistance,
                            cloudprobthreshold=cloudprobthreshold,
                            nirsnowthreshold=nirsnowthreshold,
                            greensnowthreshold=greensnowthreshold))
            print(r.stderr)
            # assert(r.stderr == b'' and r.returncode == 0)
