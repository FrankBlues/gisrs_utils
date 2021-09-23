# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:18:29 2021

@author: DELL
"""
import os
import json
from xml.etree import ElementTree as ET
import glob
import random

import numpy as np
import pandas as pd
import scipy.io as io

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import rasterio


def read_pic_arr(pic):
    with rasterio.open(pic) as ds:
        return ds.read()


def draw_images(pics, label='label', out_pic='out.png'):
    """将列表中的图片并排显示,并在下方添加label."""
    fig = plt.figure()
    n_pic = len(pics)
    for i in range(n_pic):
        ax1 = fig.add_subplot(1, n_pic, i+1)
        # 读数据,变换
        ds = rasterio.open(pics[i])
        r, c, b = ds.height, ds.width, ds.count
        img1 = ds.read()
        if str(ds.interleaving) != 'Interleaving.band':

            img = np.zeros((r, c, 3), dtype=img1.dtype)  # 强制3波段
            if str(ds.interleaving) == 'Interleaving.pixel':
                print("Change interleaving from BIP to BSQ.")
                for i in range(3):
                    img[:, :, i] = img1[i]
        else:
            img = img1
        ax1.imshow(img)
        ax1.axis('off')
    plt.subplots_adjust(wspace=0.0618, hspace=0)
    fig.text(0.5, 0.09, label, horizontalalignment='center', fontsize=23)
    # 0.06(2) 0.2(3)
    fig.tight_layout()
    plt.savefig(out_pic)


def draw_rec(img, out_img='rec.png',
             recs=[[0, 0, 64, 64]],
             R=255, G=0, B=0,
             width=1):
    """

    recs (list):list of list(xmin, ymin, xmax, ymax);
    """
    image = Image.open(img)
    # 如果不是RGB模式，转换
    if image.mode != 'RGB':
        image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for rec in recs:
        draw.rectangle(rec, outline=(R, G, B), width=width)
    image.save(out_img)
    # image.show()


def draw_polygon(img, out_img='rec.png',
                 polygons=[[0, 0, 64, 64], [0, 0, 0, 0]],
                 R=255, G=0, B=0,
                 width=1):
    """

    recs (list):list of list(xmin, ymin, xmax, ymax);
    """
    image = Image.open(img)
    # 如果不是RGB模式，转换
    if image.mode != 'RGB':
        image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for pol in polygons:
        draw.polygon(pol, outline=(R, G, B))
    image.save(out_img)


def crowdAI_show(TRAIN_IMAGES_DIRECTORY=r"D:\work\data\训练数据\遥感目标检测\crowdai\train\train\images",
                 TRAIN_ANNOTATIONS_PATH=r"D:\work\data\训练数据\遥感目标检测\crowdai\train\train\annotation-small.json"):
    from pycocotools.coco import COCO
    import skimage.io as io
    import matplotlib.pyplot as plt
    import random
    import pylab
    fig = plt.figure()
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # TRAIN_ANNOTATIONS_PATH = r"D:\work\data\训练数据\遥感目标检测\crowdai\train\train\annotation.json"
    # TRAIN_ANNOTATIONS_SMALL_PATH =
    coco = COCO(TRAIN_ANNOTATIONS_PATH)
    category_ids = coco.loadCats(coco.getCatIds())
    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    random_image_id = random.choice(image_ids)
    img = coco.loadImgs(random_image_id)[0]
    image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img["file_name"])
    I = io.imread(image_path)
    annotation_ids = coco.getAnnIds(imgIds=img['id'])
    annotations = coco.loadAnns(annotation_ids)
    # load and render the image
    plt.imshow(I)
    plt.axis('off')
    # Render annotations on top of the image
    coco.showAnns(annotations)
    plt.savefig(rf'D:\test\draw_pic\{random_image_id}.png', bbox_inches='tight',
                pad_inches=0)


def bytscl(argArry, maxValue=None, minValue=None, nodata=None, top=255):
    """将原数组指定范围(minValue ≤ x ≤ maxValue)数据拉伸至指定整型范围(0 ≤ x ≤ Top),
    输出数组类型为无符号8位整型数组.

    Note:
        Dtype of the output array is uint8.

    Args:
        argArry (numpy ndarray): 输入数组.
        maxValue (float): 最大值.默认为输入数组最大值.
        minValue (float): 最小值.默认为输入数组最大值.
        nodata (float or None): 空值，默认None，计算时排除.
        top (float): 输出数组最大值，默认255.

    Returns:
        Numpy ndarray: 线性拉伸后的数组.

    Raises:
        ValueError: If the maxValue less than or equal to the minValue.

    """
    mask = (argArry == nodata)
    retArry = np.ma.masked_where(mask, argArry)

    if maxValue is None:
        maxValue = np.ma.max(retArry)
    if minValue is None:
        minValue = np.ma.min(retArry)

    if maxValue <= minValue:
        raise ValueError("Max value must be greater than min value! ")

    retArry = (retArry - minValue) * float(top) / (maxValue - minValue)

    retArry[argArry < minValue] = 0
    retArry[argArry > maxValue] = top
    retArry = np.ma.filled(retArry, 0)
    return retArry.astype('uint8')


def linear_stretch(argArry, percent=2, leftPercent=None,
                   rightPercent=None, nodata=None,
                   only_minmax=False):
    """指定百分比对数据进行线性拉伸处理.

    Args:
        argArry (numpy ndarray): 输入图像数组.
        percent (float): 最大最小部分不参与拉伸的百分比.
        leftPercent (float):  左侧（小）不参与拉伸的百分比.
        rightPercent (float):  右侧（大）不参与拉伸的百分比.
        nodata (same as input array): 空值，默认None，计算时排除.
        only_minmax (bool): 是否只计算最大最小值

    Returns:
        numpy ndarray: 拉伸后八位无符号整型数组(0-255).

    Raises:
        ValueError: If only one of the leftPercent or the rightPercent is set.

    """
    if percent is not None:
        leftPercent = percent
        rightPercent = percent
    elif (leftPercent is None or rightPercent is None):
        raise ValueError('Wrong parameter! Both left and right percent '
                         'should be set.')

    retArry = argArry[argArry != nodata]

    minValue = np.percentile(retArry, leftPercent, interpolation="nearest")
    maxValue = np.percentile(retArry, 100 - rightPercent,
                             interpolation="nearest")

    if only_minmax:
        return minValue, maxValue
    else:
        return bytscl(argArry, maxValue=maxValue, minValue=minValue, nodata=nodata)


def gamma(image, gamma=1.0):
    """ Apply gamma correction to the channels of the image.

    Note:
        Only apply to 8 bit unsighn image.

    Args:
        image (numpy ndarray): The image array.
        gamma (float): The gamma value.

    Returns:
        Numpy ndarray: Gamma corrected image array.

    Raises:
        ValueError: If gamma value less than 0 or is nan.

    """
    if gamma <= 0 or np.isnan(gamma):
        raise ValueError("gamma must be greater than 0")

    norm = image/256.
    norm **= 1.0 / gamma
    return (norm * 255).astype('uint8')


def rle2bbox_airbus(rle, shape):
    """解析RLE格式标注"""
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))
    a[:,0] -= 1

    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)

    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)

    if x1 > shape[1]:
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    xc = (x0+x1)/(2*768)
    yc = (y0+y1)/(2*768)
    w = np.abs(x1-x0)/768
    h = np.abs(y1-y0)/768
    return [xc, yc, h, w]


if __name__ == '__main__':

    # draw ship label
    # png = r'D:\work\data\训练数据\遥感影像场景分类\MASATI-v2\multi\y0055.png'
    # xml = r'D:\work\data\训练数据\遥感影像场景分类\MASATI-v2\multi_labels\y0055.xml'
    # tree = ET.parse(xml)
    # root = tree.getroot()

    # TAS
    # name = 'traffic_2026'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\TAS\example\Data\Groundtruth\cars\{name}.gt'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\TAS\example\Data\Images\{name}.jpg'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         extent = li.rstrip('\n').split(' ')
    #         while '' in extent:
    #             extent.remove('')
    #         bbox = [float(i) for i in extent]
    #         bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes)


    # VHR-10
    # name = '633'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\NWPU VHR-10 dataset\NWPU VHR-10 dataset\ground truth\{name}.txt'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\NWPU VHR-10 dataset\NWPU VHR-10 dataset\positive image set\{name}.jpg'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         if li == '\n':
    #             continue
    #         content = li.split(',')[:4]
    #         bbox = []
    #         for e in content:
    #             if '(' in e:
    #                 e = e.lstrip('(')
    #             elif ')' in e:
    #                 e = e.strip(')')
    #             bbox.append(int(e))
    #         bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes, width=3)


    # VEDAI
    # name = '00000924'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\VEDAI\Annotations1024\{name}.txt'
    # pic_co = rf'D:\work\data\训练数据\遥感目标检测\VEDAI\Vehicules1024\{name}_co.png'
    # pic_ir = rf'D:\work\data\训练数据\遥感目标检测\VEDAI\Vehicules1024\{name}_ir.png'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         if li == '\n':
    #             continue
    #         content = li.rstrip('\n').split(' ')[-8:]
    #         bbox = [float(content[i]) for i in [0, 4, 1, 5, 2, 6, 3, 7]]
    #         bboxes.append(bbox)
    # draw_polygon(pic_co, f'D:/test/draw_pic/{name}_co.jpg', bboxes)


    # RSOD
    # name = 'playground_440'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\RSOD-Dataset\playground\Annotation\labels\{name}.txt'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\RSOD-Dataset\playground\JPEGImages\{name}.jpg'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         if li == '\n':
    #             continue
    #         extent = li.rstrip('\n').split('\t')[-4:]
    #         while '' in extent:
    #             extent.remove('')
    #         bbox = [float(i) for i in extent]
    #         bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes, width=3)


    # LEVIR
    # name = '14795'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\LEVIR\imageWithLabel\{name}.txt'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\LEVIR\imageWithLabel\{name}.jpg'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         if li == '\n':
    #             continue
    #         extent = li.rstrip('\n').split(' ')
    #         while '' in extent:
    #             extent.remove('')
    #         bbox = [float(i) for i in extent[-4:]]
    #         bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes, width=3)


    # image = Image.open(img)
    # draw = ImageDraw.Draw(image)
    # for rec in recs:
    #     draw.rectangle(rec, outline=(R, G, B), width=width)
    # image.save(out_img)


    # VisDrone
    # name = '0000284_04001_d_0000692'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\VisDrone\VisDrone2019-DET-train\annotations\{name}.txt'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\VisDrone\VisDrone2019-DET-train\images\{name}.jpg'
    # out_pic = f'D:/test/draw_pic/{name}.jpg'

    # image = Image.open(pic)
    # draw = ImageDraw.Draw(image)
    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():

    #         if li == '\n':
    #             continue

    #         extent = li.rstrip('\n').split(',')
    #         while '' in extent:
    #             extent.remove('')
    #         class_id = int(extent[5])
    #         bbox = [float(i) for i in extent[:4]]
    #         bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    #         if class_id == 0:
    #             draw.rectangle(bbox, outline=(128, 0, 0), width=1)
    #         elif class_id == 1:
    #             draw.rectangle(bbox, outline=(250, 2, 2), width=1)
    #         elif class_id == 2:
    #             draw.rectangle(bbox, outline=(0, 128, 0), width=1)
    #         elif class_id == 3:
    #             draw.rectangle(bbox, outline=(0, 255, 0), width=1)
    #         elif class_id == 4:
    #             draw.rectangle(bbox, outline=(255, 255, 0), width=1)
    #         elif class_id == 5:
    #             draw.rectangle(bbox, outline=(0, 255, 255), width=1)
    #         elif class_id == 6:
    #             draw.rectangle(bbox, outline=(255, 0, 255), width=1)
    #         elif class_id == 7:
    #             draw.rectangle(bbox, outline=(0, 0, 255), width=1)
    #         elif class_id == 8:
    #             draw.rectangle(bbox, outline=(0, 0, 128), width=1)
    #         elif class_id == 9:
    #             draw.rectangle(bbox, outline=(128, 128, 255), width=1)
    #         elif class_id == 10:
    #             draw.rectangle(bbox, outline=(0, 0, 0), width=1)
    # image.save(out_pic)


    # bboxes = []
    # for obj in root.findall('object'):
    #     bbox=[]
    #     bndbox = obj.find('bndbox')
    #     for coor in ['xmin', 'ymin', 'xmax', 'ymax']:
    #         bbox.append(int(bndbox.find(coor).text))
    #     bboxes.append(bbox)
    # draw_rec(png, 'D:/test/draw_pic/multi_y0055.png', bboxes)

    # draw pics in a row.
    # pic_dir = r'D:\work\data\训练数据\遥感影像场景分类\MLRSNet\Images'
    # for root, dirs, _ in os.walk(pic_dir):
    #     for adir in dirs:
    #         print(adir)
    #         dir1 = os.path.join(root, adir)
    #         files = os.listdir(dir1)
    #         if os.path.isfile(os.path.join(dir1, files[0])):
    #             chosen_pics = [os.path.join(dir1, f) for f in random.sample(files, 2)]
    #             # print(chosen_pics)
    #             draw_images(chosen_pics, adir, 'D:/test/draw_pic/{}.png'.format(adir))

    # GID scene class
    # pic_dir = r'D:\work\data\训练数据\遥感影像场景分类\GID\secenClass training set\RGB_15_train'
    # for root, dirs, _ in os.walk(pic_dir):
    #     for adir in dirs:
    #         print(adir)
    #         dir1 = os.path.join(root, adir)
    #         files = os.listdir(dir1)
    #         if os.path.isfile(os.path.join(dir1, files[0])):
    #             chosen_pics = [os.path.join(dir1, f) for f in random.sample(files, 3)]
    #             # print(chosen_pics)
    #             draw_images(chosen_pics, adir, 'D:/test/draw_pic/{}.png'.format(adir))

    # ITCVD
    # name = '00219'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\ITCVD\ITC_VD_Training_Testing_set\Training\GT\{name}.mat'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\ITCVD\ITC_VD_Training_Testing_set\Training\Image\{name}.jpg'


    # import scipy.io as io

    # matr = io.loadmat(gth)
    # label = matr['x'+name]

    # bboxes = []
    # for obj in label:
    #     bbox = list(obj[:4])
    #     bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes, width=3)

    # CARPK
    # name = '3_Cloudy'
    # gth = rf'D:\g_tiles\datasets\PUCPR+_devkit\data\Annotations\{name}.txt'
    # pic = rf'D:\g_tiles\datasets\PUCPR+_devkit\data\Images\{name}.jpg'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         if li == '\n':
    #             continue
    #         extent = li.rstrip('\n').split(' ')
    #         while '' in extent:
    #             extent.remove('')
    #         bbox = [float(i) for i in extent[:4]]
    #         bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes, width=3)

    # crowdAI
    # for i in range(10):
    #     crowdAI_show()

    # HRRSD
    # 每一类随机选2张
    # pics = list(range(1, 7001))
    # cat = 'parking lot'
    # random.shuffle(pics)
    # count = 0
    # for i in pics:
    #     name = f'{i:05d}'
    #     # name = '03650'
    #     pic = rf'D:\work\data\训练数据\遥感目标检测\HRRSD\Dataset-OPT2017-JPEGImages\JPEGImages-1\{name}.jpg'
    #     xml = rf'D:\work\data\训练数据\遥感目标检测\HRRSD\TGRS-HRRSD-Dataset\OPT2017\Annotations\{name}.xml'
    #     tree = ET.parse(xml)
    #     root = tree.getroot()

    #     bboxes = []
    #     for obj in root.iter('object'):
    #         _cat = obj.find('name').text
    #         if _cat != cat:
    #             continue
    #         bbox = []
    #         for o in obj.iter():
    #             if o.tag in ['xmin', 'ymin', 'xmax', 'ymax']:
    #                 bbox.append(int(o.text))
    #         bboxes.append(bbox)
    #     if len(bboxes) == 0:
    #         continue
    #     count += 1
    #     if count > 2:
    #         break
    #     draw_rec(pic, f'D:/test/draw_pic/{cat}_{name}.jpg', bboxes, width=3)

    # DIOR
    # List all categories..
    # import glob
    # cats = []
    # for xml in glob.glob(os.path.join(r'D:\work\data\训练数据\遥感目标检测\DIOR\Annotations', '*.xml')):
    #     tree = ET.parse(xml)
    #     root = tree.getroot()
    #     for obj in root.iter('object'):
    #         _cat = obj.find('name').text
    #         if _cat not in cats:
    #             cats.append(_cat)
    #     if len(cats) == 20:
    #         break

    # for cat in cats:
    #     pics = list(range(1, 11726))
    #     # cat = 'golffield'
    #     random.shuffle(pics)
    #     count = 0
    #     for i in pics:
    #         name = f'{i:05d}'
    #         # name = '03650'
    #         pic = rf'D:\work\data\训练数据\遥感目标检测\DIOR\JPEGImages-trainval\{name}.jpg'
    #         xml = rf'D:\work\data\训练数据\遥感目标检测\DIOR\Annotations\{name}.xml'
    #         tree = ET.parse(xml)
    #         root = tree.getroot()

    #         bboxes = []
    #         for obj in root.iter('object'):
    #             _cat = obj.find('name').text
    #             if _cat != cat:
    #                 continue
    #             bbox = []
    #             for o in obj.iter():
    #                 if o.tag in ['xmin', 'ymin', 'xmax', 'ymax']:
    #                     bbox.append(int(o.text))
    #             bboxes.append(bbox)
    #         if len(bboxes) == 0:
    #             continue
    #         count += 1
    #         if count > 2:
    #             break
    #         draw_rec(pic, f'D:/test/draw_pic/{cat}_{name}.jpg', bboxes, width=3)

    # SAR ship
    # name = 'newship011017035'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\SAR-Ship-Dataset\ship_dataset_v0\{name}.txt'
    # pic = rf'D:\work\data\训练数据\遥感目标检测\SAR-Ship-Dataset\ship_dataset_v0\{name}.jpg'

    # bboxes = []
    # with open(gth) as inf:
    #     for li in inf.readlines():
    #         if li == '\n':
    #             continue
    #         extent = li.rstrip('\n').split(' ')
    #         while '' in extent:
    #             extent.remove('')
    #         mid_x, mid_y, w, h = [float(i)*256 for i in extent[-4:]]
    #         bbox = [mid_x - w/2, mid_y - h/2, mid_x + w/2, mid_y + h/2]
    #         bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/{name}.jpg', bboxes, width=3)

    # AIR-SARShip
    # name = '83'
    # gth = rf'D:\work\data\训练数据\遥感目标检测\AIR-SARShip\AIR-SARShip-2.0-xml\{name}.xml'
    # img = rf'D:\work\data\训练数据\遥感目标检测\AIR-SARShip\AIR-SARShip-2.0-data\{name}.tiff'
    # out = f'D:/test/draw_pic/{name}.png'
    # with rasterio.open(img) as ds:
    #     kargs = ds.meta.copy()
    #     data = ds.read(1)
    #     db = np.log10(data)
    #     strech_data = linear_stretch(db, percent=1)
    #     kargs.update({'dtype': 'uint8',
    #                   'driver': 'PNG'})

    #     with rasterio.open(out, 'w', **kargs) as dst:
    #         dst.write(strech_data, 1)

    # tree = ET.parse(gth)
    # root = tree.getroot()
    # bboxes = []
    # for obj in root.iter('object'):
    #     lx = []
    #     ly = []
    #     for o in obj.iter('point'):
    #         x, y = o.text.split(', ')
    #         lx.append(int(x))
    #         ly.append(int(y))
    #     bbox = [min(lx), min(ly), max(lx), max(ly)]
    #     bboxes.append(bbox)
    # draw_rec(out, f'D:/test/draw_pic/{name}_label.png', bboxes, width=3)

    # HRSID
    # TRAIN_IMAGES_DIRECTORY = r"D:\work\data\训练数据\遥感目标检测\HRSID\HRSID_JPG\JPEGImages"
    # TRAIN_ANNOTATIONS_PATH = r"D:\work\data\训练数据\遥感目标检测\HRSID\HRSID_JPG\annotations\train2017.json"
    # crowdAI_show(TRAIN_IMAGES_DIRECTORY, TRAIN_ANNOTATIONS_PATH)

    # airbus ship 参考kaggle用户提交
    # import pandas as pd
    # ships = pd.read_csv(r"G:\遥感数据集\遥感目标检测\airbus-ship-detection\train_ship_segmentations_v2.csv")
    # path =r"G:/遥感数据集/遥感目标检测/airbus-ship-detection/train_v2/"

    # # 添加字段 统计影像船只个数
    # ships["Ship"] = ships["EncodedPixels"].map(lambda x:1 if isinstance(x,str) else 0)
    # ship_unique = ships[["ImageId","Ship"]].groupby("ImageId").agg({"Ship":"sum"}).reset_index()

    # # 解析RLE格式为外边框,替换原标注字段
    # ships["Boundingbox"] = ships["EncodedPixels"].apply(lambda x:rle2bbox_airbus(x,(768,768)) if isinstance(x,str) else np.NaN)
    # ships.drop("EncodedPixels", axis =1, inplace =True)

    # # 统计像素面积，筛除掉面积小的船只
    # ships["BoundingboxArea"]=ships["Boundingbox"].map(lambda x:x[2]*768*x[3]*768 if x==x else 0)
    # ships = ships[ships["BoundingboxArea"]>np.percentile(ships["BoundingboxArea"],1)]

    # # 平衡数据 不同数量船只的图像各取最大1000个样本
    # balanced_df = ship_unique.groupby("Ship").apply(lambda x:x.sample(1000) if len(x)>=1000 else x.sample(len(x)))
    # balanced_df.reset_index(drop=True,inplace=True)

    # # 合并
    # balanced_bbox = ships.merge(balanced_df[["ImageId"]], how ="inner", on = "ImageId")

    # plt.figure(figsize =(20,20))
    # for i in range(16):
    #     imageid = balanced_df[balanced_df.Ship ==i].iloc[1][0]
    #     # image = np.array(cv2.imread(path+imageid)[:,:,::-1])
    #     image = Image.open(path+imageid)
    #     draw = ImageDraw.Draw(image)
    #     if i>0:
    #         bbox = balanced_bbox[balanced_bbox.ImageId==imageid]["Boundingbox"]

    #         for items in bbox:
    #             Xmin  = int((items[0]-items[3]/2)*768)
    #             Ymin  = int((items[1]-items[2]/2)*768)
    #             Xmax  = int((items[0]+items[3]/2)*768)
    #             Ymax  = int((items[1]+items[2]/2)*768)
    #             draw.rectangle((Xmin, Ymin, Xmax, Ymax), outline=(255, 0, 0), width=2)

    #     plt.subplot(4,4,i+1)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.imshow(image)
    #     # plt.title("Bulunan gemi sayısı = {}".format(i))
    # plt.savefig('d:/test/draw_pic/airbus_ship.png')

    # KSC
    # img = r'D:\work\data\训练数据\语义分割\Kennedy Space Center\KSC.mat'
    # gth = r'D:\work\data\训练数据\语义分割\Kennedy Space Center\KSC_gt.mat'


    # matr = io.loadmat(gth)
    # gth_data = matr.get('KSC_gt')

    # matr = io.loadmat(img)
    # img_data = matr.get('KSC')
    # # 标注数据
    # kargs = {'driver': 'GTiff',
    #          'count': 1,
    #          'dtype': 'uint8',
    #          'height': 512,
    #          'width': 614}
    # with rasterio.open('d:/test.tif', 'w', **kargs) as dst:
    #     dst.write(gth_data, 1)

    # # 取其中3个波段进行合成
    # kargs = {'driver': 'GTiff',
    #          'count': 3,
    #          'dtype': 'uint16',
    #          'height': 512,
    #          'width': 614}
    # with rasterio.open('d:/test_rgb.tif', 'w', **kargs) as dst:
    #     dst.write(img_data[:, :, 28], 1)
    #     dst.write(img_data[:, :, 13], 2)
    #     dst.write(img_data[:, :, 6], 3)

    # Botswana
    # img = r'D:\work\data\训练数据\语义分割\Botswana\Botswana.mat'
    # gth = r'D:\work\data\训练数据\语义分割\Botswana\Botswana_gt.mat'
    # matr = io.loadmat(gth)
    # gth_data = matr.get('Botswana_gt')

    # matr = io.loadmat(img)
    # img_data = matr.get('Botswana')
    # # 标注数据
    # kargs = {'driver': 'GTiff',
    #           'count': 1,
    #           'dtype': 'uint8',
    #           'height': 1476,
    #           'width': 256}
    # with rasterio.open('d:/test_bots.tif', 'w', **kargs) as dst:
    #     dst.write(gth_data, 1)

    # # 取其中3个波段进行合成
    # kargs = {'driver': 'GTiff',
    #           'count': 3,
    #           'dtype': 'uint16',
    #           'height': 1476,
    #           'width': 256}
    # with rasterio.open('d:/test_bots_rgb.tif', 'w', **kargs) as dst:
    #     dst.write(img_data[:, :, 18], 1)
    #     dst.write(img_data[:, :, 5], 2)
    #     dst.write(img_data[:, :, 0], 3)


    # Salinas
    # img = r'D:\work\data\训练数据\语义分割\Salinas\Salinas Scene\Salinas_corrected.mat'
    # gth = r'D:\work\data\训练数据\语义分割\Salinas\Salinas Scene\Salinas_gt.mat'
    # matr = io.loadmat(gth)
    # gth_data = matr.get('salinas_gt')

    # matr = io.loadmat(img)
    # img_data = matr.get('salinas_corrected')
    # # 标注数据
    # kargs = {'driver': 'GTiff',
    #           'count': 1,
    #           'dtype': 'uint8',
    #           'height': 512,
    #           'width': 217}
    # with rasterio.open('d:/test_Salinas.tif', 'w', **kargs) as dst:
    #     dst.write(gth_data, 1)

    # # 取其中3个波段进行合成
    # kargs = {'driver': 'GTiff',
    #           'count': 3,
    #           'dtype': 'int16',
    #           'height': 512,
    #           'width': 217}
    # with rasterio.open('d:/test_Salinas_rgb.tif', 'w', **kargs) as dst:
    #     dst.write(img_data[:, :, 28], 1)
    #     dst.write(img_data[:, :, 13], 2)
    #     dst.write(img_data[:, :, 7], 3)


    # Pavia
    # img = r'd:\work\data\训练数据\语义分割\Pavia Centre\Pavia.mat'
    # gth = r'd:\work\data\训练数据\语义分割\Pavia Centre\Pavia_gt.mat'

    # img = r'D:\work\data\训练数据\语义分割\Pavia University\PaviaU.mat'
    # gth = r'D:\work\data\训练数据\语义分割\Pavia University\PaviaU_gt.mat'

    # matr = io.loadmat(gth)
    # gth_data = matr.get('paviaU_gt')

    # matr = io.loadmat(img)
    # img_data = matr.get('paviaU')
    # # 标注数据
    # kargs = {'driver': 'GTiff',
    #           'count': 1,
    #           'dtype': 'uint8',
    #           'height': 610,  # 1096,
    #           'width': 340  # 715
    #           }
    # with rasterio.open('d:/test_PaviaU.tif', 'w', **kargs) as dst:
    #     dst.write(gth_data, 1)

    # # 取其中3个波段进行合成
    # kargs = {'driver': 'GTiff',
    #           'count': 3,
    #           'dtype': 'uint16',
    #           'height': 610, # 1096,
    #           'width': 340 # 715
    #           }
    # with rasterio.open('d:/test_PaviaU_rgb.tif', 'w', **kargs) as dst:
    #     dst.write(img_data[:, :, 62], 1)
    #     dst.write(img_data[:, :, 22], 2)
    #     dst.write(img_data[:, :, 6], 3)

    # Zurish
    # name = 'zh1'
    # img = rf'D:\work\data\训练数据\语义分割\Zurich_dataset_v1.0\images_tif\{name}.tif'
    # gth = rf'D:\work\data\训练数据\语义分割\Zurich_dataset_v1.0\groundtruth\{name}_GT.tif'

    # out = f'D:/test/draw_pic/{name}_5.tif'
    # with rasterio.open(img) as ds:
    #     kargs = ds.meta.copy()

    #     kargs.update({'dtype': 'uint8',
    #                   'count': 3,
    #                   'driver': 'GTiff'})

    #     mins = []
    #     maxs = []
    #     rgbs = []
    #     for b in range(3, 0, -1):
    #         data = ds.read(b)
    #         rgbs.append(data)

    #         minv, maxv = linear_stretch(data, percent=0.25, only_minmax=True)
    #         mins.append(minv)
    #         maxs.append(maxv)

    #     bands = len(mins)

    #     gammas = []
    #     streched_data = []
    #     for data in rgbs:
    #         strech_data = bytscl(data, maxValue=sum(maxs)/bands,
    #                              minValue=sum(mins)/bands)
    #         streched_data.append(strech_data)
    #         gammav = np.log10(0.5)/np.log10(strech_data.mean()/256)
    #         print("gamma:{}".format(1/gammav))
    #         gammas.append(1/gammav)
    #     rgbs = None

    #     with rasterio.open(out, 'w', **kargs) as dst:
    #         for i, strech_data in enumerate(streched_data):
    #             g_data = gamma(strech_data, min(gammas))

    #             dst.write(g_data, i + 1)

    # RIT-18
    from scipy.io import loadmat

    file_path = r'D:\work\data\训练数据\语义分割\RIT-18\rit18_data.mat'

    # dataset = loadmat(file_path)

    # def write_img(out, data):
    #     kargs = {'driver': 'GTiff',
    #              'count': 6,
    #              'dtype': 'uint16',
    #              'height': 9393,  # 1096,
    #              'width': 5642  # 715
    #              }
    #     with rasterio.open(out, 'w', **kargs) as dst:
    #         for i in range(6):
    #             dst.write(data[i, :, :], i + 1)

    # def write_label(out, data):
    #     kargs = {'driver': 'GTiff',
    #              'count': 1,
    #              'dtype': 'uint8',
    #              'height': 9393,
    #              'width': 5642
    #              }
    #     with rasterio.open(out, 'w', **kargs) as dst:
    #         dst.write(data, 1)

    #Load Training Data and Labels
    # train_data = dataset['train_data']
    # train_mask = train_data[-1]
    # train_data = train_data[:6]
    # train_labels = dataset['train_labels']

    # # 写出数据
    # write_img('d:/rit18_train_data.tif', train_data)

    # # 写出标签
    # write_label('d:/rit18_train_labels.tif', train_labels)

    # #Load Validation Data and Labels
    # val_data = dataset['val_data']
    # val_mask = val_data[-1]
    # val_data = val_data[:6]
    # val_labels = dataset['val_labels']

    # # 写出数据
    # write_img('d:/rit18_val_data.tif', val_data)

    # # 写出标签
    # write_label('d:/rit18_val_labels.tif', val_labels)


    #Load Test Data
    # test_data = dataset['test_data']
    # test_mask = test_data[-1]
    # test_data = test_data[:6]
    # # 写出数据
    # write_img('d:/rit18_test_data.tif', test_data)

    # band_centers = dataset['band_centers'][0]
    # band_center_units = dataset['band_center_units']
    # classes = dataset['classes']

    # #Print some info about the dataset
    # print(dataset['sensor'][0])
    # print(dataset['info'][0])

    # Dstl-SIFD
    # 给图像加上坐标（非地理）
    # name = '6010_1_2'
    # img = rf'D:\work\data\训练数据\语义分割\Dstl-SIFD\three_band\{name}.tif'
    # coor = r'd:\work\data\训练数据\语义分割\Dstl-SIFD\grid_sizes.csv'
    # coor_ds = pd.read_csv(coor).set_index('Unnamed: 0')
    # xmax, ymin = coor_ds.loc[name]
    
    # with rasterio.open(img) as src:
    #     kargs = src.meta.copy()
    #     trans = kargs.get('transform')
    #     gdal_trans = (0, xmax/src.width, 0.0, 0, 0.0, ymin/src.height)
    #     new_trans = trans.from_gdal(*gdal_trans)
    #     kargs.update({'transform': new_trans})
        
    #     with rasterio.open(f'd:/temp11/{name}.tif', 'w', **kargs) as dst:
    #         dst.write(src.read())
    # # geojson转换为shapefile，在gis软件中制作示例
    # label_dir = rf'D:\work\data\训练数据\语义分割\Dstl-SIFD\train_geojson_v3\{name}'
    # for label in glob.glob(os.path.join(label_dir, '*.geojson')):
    #     basename = os.path.basename(label)
    #     dst = os.path.join('d:/temp11', basename.replace('.geojson', '.shp'))
    #     cmd = f"ogr2ogr {dst} {label}"
    #     os.system(cmd)

    # SLOVENIA
    # https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html
    # from eolearn.core import EOPatch

    # # patch = EOPatch()
    # eopatch = EOPatch.load(r'D:\work\data\训练数据\语义分割\Slovenia\eopatch_id_0_col_0_row_19')
    
    # fig, ax = plt.subplots(figsize=(4, 4))
    # # ax = axs[0][0]
    # ax.imshow(np.clip(eopatch.data['BANDS'][0][..., [2, 1, 0]] * 3.5, 0, 1))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_aspect("auto")
    
    # plt.tight_layout()
    
    # So2Sat
    # select one of the given files
    import h5py
    fileOfChoice = r'D:\work\data\训练数据\语义分割\So2Sat LCZ42\validation.h5'
    
    # show the variables in selected file
    # fid = h5py.File(fileOfChoice, 'r')
    # print('INFO:    The names of variables in the file of \''+fileOfChoice+'\':')
    # #print fid.keys()
    
    # # load the data into memory
    # print('INFO:    Loading sentinel-1 data patches ...')
    # s1 = np.array(fid['sen1'])
    # print('INFO:    Sentinel-1 data dimension:')
    # print(s1.shape)

    # print('INFO:    Loading sentinel-2 data patches ...')
    # s2 = np.array(fid['sen2'])
    # print('INFO:    Sentinel-2 data dimension:')
    # print(s2.shape)

    # print('INFO:    Loading label ...')
    # lab = np.array(fid['label'])
    # print('INFO:    Label dimension:')
    # print(lab.shape)

    # # plt.axis('off')
    # import random
    # idx = 0
    # for _ in range(200):
    #     idx = random.choice(range(24119))
    
    #     idx_lab = np.argmax(lab[idx])
    #     print(f'Index:{idx};Label: {idx_lab}')
        
    #     if idx_lab != 14:
    #         continue

    #     # visualization, plot the first pair of Sentinel-1 and Sentinel-2 patches of training.h5
    #     plt.subplot(121)
    #     plt.imshow(10*np.log10(s1[idx,:,:,4]),cmap=plt.cm.get_cmap('gray'));
    #     # plt.colorbar()
    #     plt.title('Sentinel-1')
    #     ax = plt.gca()
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    
    #     plt.subplot(122)
    #     plt.imshow(s2[idx,:,:,1],cmap=plt.cm.get_cmap('gray'))
    #     # plt.colorbar()
    #     plt.title('Sentinel-2')
    #     ax = plt.gca()
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    
    #     plt.subplots_adjust(wspace=0.0618, hspace=0)
    #     plt.tight_layout()
    #     plt.savefig(f"d:/temp11/{idx_lab}_so2sat_{idx}.png")
    
    # landcover.ai
    # lcai = r'D:\work\data\训练数据\语义分割\landcover-ai\landcover.ai\images'
    # for f in glob.glob(os.path.join(lcai, '*.tif')):
    #     with rasterio.open(f) as ds:
    #         print(ds.width, ds.height)

    # agriculture vision
    # 归并文件夹
    # ref = r'E:\Agriculture-Vision\地址.txt'
    # all_dic = {}
    # with open(ref) as f:
    #     for line in f.readlines():
    #         lst = line.split('/')
    #         _dir, _file = lst[-2], lst[-1].rstrip('\n')
    #         if _dir not in all_dic.keys():
    #             all_dic[_dir] = []
    #             all_dic[_dir].append(_file)
    #         else:
    #             all_dic[_dir].append(_file)

    # file_dir = 'E:/Agriculture-Vision/raw/'
    # os.chdir(file_dir)
    # import shutil
    # files = os.listdir(file_dir)
    # for ff in files:
    #     if os.path.isfile(ff):
    #         print(f"处理中:{ff};")
    #         for k in all_dic.keys():
    #             if ff in all_dic[k]:
    #                 if not os.path.isdir(k):
    #                     os.mkdir(k)
    #                     shutil.move(ff, k)
    #                 else:
    #                     shutil.move(ff, k)

    # GETNET
    # img = r'D:\work\data\训练数据\变化检测\dataset zuixin\river_after.mat'
    # gth = r'D:\work\data\训练数据\变化检测\dataset zuixin\groundtruth.mat'
    # mat_gth = io.loadmat(gth)
    # gth_data = mat_gth.get('lakelabel_v1')

    # mat_img = io.loadmat(img)
    # img_data = mat_img.get('river_after')
    # # 标注数据
    # kargs = {'driver': 'GTiff',
    #           'count': 1,
    #           'dtype': 'uint8',
    #           'height': 463,
    #           'width': 241}
    # with rasterio.open('d:/temp11/lakelabel_v1.tif', 'w', **kargs) as dst:
    #     dst.write(gth_data, 1)

    # # # 取其中3个波段进行合成
    # kargs = {'driver': 'GTiff',
    #           'count': 198,
    #           'dtype': 'int16',
    #           'height': 463,
    #           'width': 241}
    # with rasterio.open('d:/temp11/river_after.tif', 'w', **kargs) as dst:
    #     for i in range(198):
    #         dst.write(img_data[:, :, i], i + 1)

    # Hermiston
    # img = r'G:\遥感数据集\变化检测\Hermiston\ChangeDetectionDataset-master\Hermiston\hermiston2004.mat'
    # gth = r'G:\遥感数据集\变化检测\Hermiston\ChangeDetectionDataset-master\Hermiston\rdChangesHermiston_5classes.mat'
    # mat_gth = io.loadmat(gth)
    # gth_data = mat_gth.get('gt5clasesHermiston')

    # mat_img = io.loadmat(img)
    # img_data = mat_img.get('HypeRvieW')
    # # 标注数据
    # kargs = {'driver': 'GTiff',
    #           'count': 1,
    #           'dtype': 'uint8',
    #           'height': 390,
    #           'width': 200}
    # with rasterio.open('d:/temp11/gt5clasesHermiston.tif', 'w', **kargs) as dst:
    #     dst.write(gth_data, 1)

    # # # 取其中3个波段进行合成
    # kargs = {'driver': 'GTiff',
    #           'count': 242,
    #           'dtype': 'float64',
    #           'height': 390,
    #           'width': 200}
    # with rasterio.open('d:/temp11/HypeRvieWHermiston2004.tif', 'w', **kargs) as dst:
    #     for i in range(242):
    #         dst.write(img_data[:, :, i], i + 1)


    # OSV
    # 每一类随机选2张
    # pics = list(range(1, 601))
    # cat = 'traffic_sign'  # light car traffic_sign crosswark warning_line
    # random.shuffle(pics)
    # count = 0
    # for i in pics:
    #     name = f'{i:06d}'
    #     # name = '03650'
    #     pic = rf'D:\work\data\训练数据\Omnidirectional Street-view Dataset\DriscollHealy\JPEGImages\{name}.jpg'
    #     xml = rf'D:\work\data\训练数据\Omnidirectional Street-view Dataset\DriscollHealy\Annotations\{name}.xml'
    #     tree = ET.parse(xml)
    #     root = tree.getroot()

    #     bboxes = []
    #     for obj in root.iter('object'):
    #         _cat = obj.find('name').text
    #         if _cat != cat:
    #             continue
    #         bbox = []
    #         for o in obj.iter():
    #             if o.tag in ['xmin', 'ymin', 'xmax', 'ymax']:
    #                 bbox.append(int(o.text))
    #         bboxes.append(bbox)
    #     if len(bboxes) == 0:
    #         continue
    #     count += 1
    #     if count > 3:
    #         break
    #     draw_rec(pic, f'D:/test/draw_pic/{cat}_{name}.jpg', bboxes, width=3)

    # SenseEarth classify
    # jsonf = r'G:\遥感数据集\遥感影像场景分类\SenseEarth classify\class_indices.json'
    # with open(jsonf) as j:
    #     class_indices = json.load(j)
    #     classes = [s.split('/')[-1] for s in list(class_indices.keys())]
    #     print(len(classes))
    #     print(','.join(classes))
    #     # print(list(map(lambda s: s.split[-1], classes)))
    # label_txt = r'G:\遥感数据集\遥感影像场景分类\SenseEarth classify\train.txt'
    # pic_dir = 'G:/遥感数据集/遥感影像场景分类/SenseEarth classify/train/'
    # with open(label_txt) as f:
    #     lines = f.readlines()
    #     pic_all_cat = {}

    #     for l in lines:
    #         ll = l.rstrip('\n').split(' ')
    #         for i in range(51):
    #             if int(ll[-1]) == i:
    #                 try:
    #                     pic_all_cat[i].append(ll[0])
    #                 except KeyError:
    #                     pic_all_cat[i]=[]
    #                     pic_all_cat[i].append(ll[0])
    #     for i in range(51):
    #         pic_all_cat[i] = random.sample(pic_all_cat[i], 2)
    #         draw_images([pic_dir + p for p in pic_all_cat[i]], label=classes[i], out_pic='D:/test/draw_pic/' + str(i) + classes[i] + '.png')
            # break
        # print(pic_all_cat)


    # VArcGIS
    # pic_dir = r'G:\遥感数据集\遥感影像场景分类\VArcGIS\VArcGIS'
    # for root, dirs, _ in os.walk(pic_dir):
    #     for adir in dirs:
    #         print(adir)
    #         dir1 = os.path.join(root, adir)
    #         files = os.listdir(dir1)
    #         if os.path.isfile(os.path.join(dir1, files[0])):
    #             chosen_pics = [os.path.join(dir1, f) for f in random.sample(files, 2)]
    #             draw_images(chosen_pics, adir, 'D:/test/draw_pic/{}.png'.format(adir))

    # bridge
    # 每一类随机选2张
    # name = 245
    # pic = rf'G:\遥感数据集\遥感目标检测\bridges_dataset\JPEGImages\{name}.jpg'
    # xml = rf'G:\遥感数据集\遥感目标检测\bridges_dataset\Annotations\{name}.xml'
    # tree = ET.parse(xml)
    # root = tree.getroot()

    # bboxes = []
    # for obj in root.iter('object'):
    #     _cat = obj.find('name').text
    #     bbox = []
    #     for o in obj.iter():
    #         if o.tag in ['xmin', 'ymin', 'xmax', 'ymax']:
    #             bbox.append(int(o.text))
    #     bboxes.append(bbox)
    # draw_rec(pic, f'D:/test/draw_pic/bridge_{name}.jpg', bboxes, width=8)

    # image_dir = r'G:\遥感数据集\2018 Open AI Tanzania Building Footprint\image'
    # rows, cols = [], []
    # res = []
    # for img in glob.glob(os.path.join(image_dir, '*.tif')):
    #     ds = rasterio.open(img)
    #     print(ds.crs)
    #     rows.append(ds.height)
    #     cols.append(ds.width)
    #     res.append(ds.res)
    # print(min(rows), max(rows))
    # print(min(cols), max(cols))
    # print(min(res), max(res))
    
    # mini Inria Aerial Image Labeling Dataset
    train_mask = r'G:\遥感数据集\train_mask.csv'
    import pandas as pd
    df = pd.read_csv(train_mask)
    with open(train_mask) as csvf:
        lines = csvf.readlines()
    
    




