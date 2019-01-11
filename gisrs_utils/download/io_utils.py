# -*- coding: utf-8 -*-
"""
Module for local reading and writing of data
"""

import os
import csv
import json
import struct
import logging
import numpy as np
import zipfile
from xml.etree import ElementTree
try:
    import tifffile as tiff
except ImportError:
    print("tifffile module missing,install when needed")
    pass
try:
    import cv2
except ImportError:
    print("cv2 module missing,install when needed")
    pass
try:
    import gdal
except ImportError:
    print("gdal module missing,install when needed")
    pass
LOGGER = logging.getLogger(__name__)

CSV_DELIMITER = ';'


def read_tiff_image(filename):
    """ Read data from TIFF file

    :param filename: name of TIFF file to be read
    :type filename: str
    :return: data stored in TIFF file
    """
    return tiff.imread(filename)


def read_raster_gdal(filename):
    try:
        ds = gdal.Open(filename)

        return ds.ReadAsArray(), ds.GetGeoTransform(), ds.GetProjection()
    finally:
        del ds


def read_jp2_image(filename):
    """ Read data from JPEG2000 file

    :param filename: name of JPEG2000 file to be read
    :type filename: str
    :return: data stored in JPEG2000 file
    """
    # Reading with glymur library:
    # return glymur.Jp2k(filename)[:]
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    with open(filename, 'rb') as file:
        bit_depth = get_jp2_bit_depth(file)

    return _fix_jp2_image(image, bit_depth)


def read_image(filename):
    """ Read data from PNG or JPG file

    :param filename: name of PNG or JPG file to be read
    :type filename: str
    :return: data stored in JPG file
    """
    return cv2.imread(filename, cv2.IMREAD_UNCHANGED)


def read_text(filename):
    """ Read data from text file

    :param filename: name of text file to be read
    :type filename: str
    :return: data stored in text file
    """
    with open(filename, 'r') as file:
        return file.read()   # file.readline() for reading 1 line


def read_csv(filename, delimiter=CSV_DELIMITER):
    """ Read data from CSV file

    :param filename: name of CSV file to be read
    :type filename: str
    :param delimiter: type of CSV delimiter. Default is ``;``
    :type delimiter: str
    :return: data stored in CSV file as list
    """
    with open(filename, 'r') as file:
        return list(csv.reader(file, delimiter=delimiter))


def read_json(filename):
    """ Read data from JSON file

    :param filename: name of JSON file to be read
    :type filename: str
    :return: data stored in JSON file
    """
    with open(filename, 'r') as file:
        return json.load(file)


def read_xml(filename):
    """ Read data from XML or GML file

    :param filename: name of XML or GML file to be read
    :type filename: str
    :return: data stored in XML file
    """
    return ElementTree.parse(filename)


def read_numpy(filename):
    """ Read data from numpy file

    :param filename: name of numpy file to be read
    :type filename: str
    :return: data stored in file as numpy array
    """
    return np.load(filename)


def write_tiff_image(filename, image, compress=False):
    """ Write image data to TIFF file

    :param filename: name of file to write data to
    :type filename: str
    :param image: image data to write to file
    :type image: numpy array
    :param compress: whether to compress data. If ``True``, lzma compression is used. Default is ``False``
    :type compress: bool
    """
    if compress:
        # loseless compression, works very well on masks
        return tiff.imsave(filename, image, compress='lzma')
    return tiff.imsave(filename, image)


def write_tiff_image_gdal_oneband(filename, image, datatype=gdal.GDT_Byte, prj=None, geoTrans=None):
    """ Write image data to TIFF file
        geoTrans = (lon_min, res,0,lat_max,0,-res)

    """

    cols = image.shape[1]
    rows = image.shape[0]

    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(filename, cols, rows, 1,  datatype)

    if geoTrans != None:
        outdata.SetGeoTransform(geoTrans)

    # if prj != None:
    # outdata.SetProjection(prj)

    outdata.GetRasterBand(1).WriteArray(image)
    outdata = None


def write_jp2_image(filename, image):
    """ Write image data to JPEG2000 file

    :param filename: name of JPEG2000 file to write data to
    :type filename: str
    :param image: image data to write to file
    :type image: numpy array
    :return: jp2k object
    """
    # Writing with glymur library:
    # return glymur.Jp2k(filename, data=image)
    return cv2.imwrite(filename, image)


def write_text(filename, data, add=False):
    """ Write image data to text file

    :param filename: name of text file to write data to
    :type filename: str
    :param data: image data to write to text file
    :type data: numpy array
    :param add: whether to append to existing file or not. Default is ``False``
    :type add: bool
    """
    write_type = 'a' if add else 'w'
    with open(filename, write_type) as file:
        file.write(data)


def write_csv(filename, data, delimiter=CSV_DELIMITER):
    """ Write image data to CSV file

    :param filename: name of CSV file to write data to
    :type filename: str
    :param data: image data to write to CSV file
    :type data: numpy array
    :param delimiter: delimiter used in CSV file. Default is ``;``
    :type delimiter: str
    """
    with open(filename, 'w') as file:
        csv_writer = csv.writer(file, delimiter=delimiter)
        for line in data:
            csv_writer.writerow(line)


def write_csv_list(filename, list, add=False, delimiter=CSV_DELIMITER):
    """ Write image data to csv file

    :param filename: name of text file to write data to
    :type filename: str
    :param data: list to write to csv file as one line
    :type data: list
    :param add: whether to append to existing file or not. Default is ``False``
    :type add: bool
    :param delimiter: delimiter used in CSV file. Default is ``;``
    :type delimiter: str
    """
    write_type = 'a' if add else 'w'
    with open(filename, write_type,newline='') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerow(list)


def write_json(filename, data):
    """ Write data to JSON file

    :param filename: name of JSON file to write data to
    :type filename: str
    :param data: data to write to JSON file
    :type data: list, tuple
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=True)


def write_xml(filename, element_tree):
    """ Write data to XML or GML file

    :param filename: name of XML or GML file to write data to
    :type filename: str
    :param element_tree: data as ElementTree object
    :type element_tree: xmlElementTree
    """
    return element_tree.write(filename)
    # this will write declaration tag in first line:
    # return element_tree.write(filename, encoding='utf-8', xml_declaration=True)


def write_numpy(filename, data):
    """ Write data as numpy file

    :param filename: name of numpy file to write data to
    :type filename: str
    :param data: data to write to numpy file
    :type data: numpy array
    """
    return np.save(filename, data)


def get_jp2_bit_depth(stream):
    """Reads bit encoding depth of jpeg2000 file in binary stream format

    :param stream: binary stream format
    :type stream: Binary I/O (e.g. io.BytesIO, io.BufferedReader, ...)
    :return: bit depth
    :rtype: int
    """
    stream.seek(0)
    while True:
        read_buffer = stream.read(8)
        if len(read_buffer) < 8:
            raise ValueError('Image Header Box not found in Jpeg2000 file')

        _, box_id = struct.unpack('>I4s', read_buffer)

        if box_id == b'ihdr':
            read_buffer = stream.read(14)
            params = struct.unpack('>IIHBBBB', read_buffer)
            return (params[3] & 0x7f) + 1


def _fix_jp2_image(image, bit_depth):
    """Because opencv library incorrectly reads jpeg2000 images with 15-bit encoding this function corrects the
    values in image.

    :param image: image read by opencv library
    :type image: numpy array
    :param bit_depth: bit depth of jp2 image encoding
    :type bit_depth: int
    :return: corrected image
    :rtype: numpy array
    """
    if bit_depth in [8, 16]:
        return image
    if bit_depth == 15:
        return image >> 1
    raise ValueError('Bit depth {} of jp2 image is currently not supported. '
                     'Please raise an issue on package Github page'.format(bit_depth))


def zipDir(outzipfile, zipdir):
    """
    目录下所有文件压缩到指定文件
    参数：

    返回：
    """
    with zipfile.ZipFile(outzipfile, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(zipdir):
            for filename in filenames:
                pathfile = os.path.join(dirpath, filename)
                arcname = pathfile[len(os.path.realpath(zipdir)):].strip(
                    os.path.sep)
                z.write(pathfile, arcname)


if __name__ == '__main__':
    # pass
    gml = r'F:\SENTINEL\download\down0702\S2_49QCD_20180602_0\MSK_CLOUDS_B00.gml'
    refImage = r'F:\SENTINEL\图片任务组_20180524_1153\B02.jp2'
    #outDS = gdal.Open(refImage)
    #import io_utils
    #bandInfo = io_utils.read_raster_gdal(refImage)

    #geoTrans = bandInfo[1]
    #projInfo = bandInfo[2]

    #rows,cols = bandInfo[0].shape
    d = 0
    c = 0
    tree = read_xml(gml)
    root = tree.getroot()
    maskMembers = root.find('{http://www.opengis.net/eop/2.0}maskMembers')
    # print(len(maskMembers.getchildren()))
    for msk in maskMembers.iter('{http://www.opengis.net/eop/2.0}MaskFeature'):
        # print(msk.tag)
        # print(msk.getchildren())
        if msk.find('{http://www.opengis.net/eop/2.0}maskType').text == 'CIRRUS':
            # print(msk.attrib['{http://www.opengis.net/gml/3.2}id'])
            # if 'CIRRUS' in msk.attrib['{http://www.opengis.net/gml/3.2}id'] :
            # print(msk.tag)
            c += 1
            maskMembers.remove(msk)
    # print(c)
    # print(len(maskMembers.getchildren()))
    root.remove(maskMembers)
    root.append(maskMembers)

    #tree1 = ElementTree()
    # ElementTree.dump(root)

    tree.write('d:/new3.gml')


# =============================================================================
#     gg = """        <gml:Polygon gml:id="OPAQUE.5_Polygon">
#           <gml:exterior>
#             <gml:LinearRing>
#               <gml:posList srsDimension="2">877440 2600040 878520 2600040 878520 2599920 878580 2599920 878580 2599800 878640 2599800 878640 2599440 878700 2599440 878700 2599560 878820 2599560 878820 2599620 878940 2599620 878940 2599800 878880 2599800 878880 2600040 879900 2600040 879900 2599800 879840 2599800 879840 2599680 879780 2599680 879780 2599560 879660 2599560 879660 2599440 879720 2599440 879720 2599320 879780 2599320 879780 2599200 879840 2599200 879840 2598840 879780 2598840 879780 2598720 879720 2598720 879720 2598600 879600 2598600 879600 2598540 879480 2598540 879480 2598480 879360 2598480 879360 2598420 879060 2598420 879060 2598480 878940 2598480 878940 2598540 878820 2598540 878820 2598600 878760 2598600 878760 2598660 878700 2598660 878700 2598780 878640 2598780 878640 2598900 878580 2598900 878580 2599200 878520 2599200 878520 2599080 878400 2599080 878400 2599020 878280 2599020 878280 2598960 877860 2598960 877860 2599020 877740 2599020 877740 2599080 877620 2599080 877620 2599140 877560 2599140 877560 2599260 877500 2599260 877500 2599380 877440 2599380 877440 2600040</gml:posList>
#             </gml:LinearRing>
#           </gml:exterior>
#         </gml:Polygon>"""
# =============================================================================

    #from osgeo import ogr
    #point = ogr.CreateGeometryFromGML(gg)

    #tt = '<a>55ddf</a>sfdsf<a>55ddffdf</a>edfg<a>55ddf</a>'
    import re

    txt = read_text(gml)
    # print(txt)
    #ps = re.findall(r'<gml:Polygon[\s\S]*?</gml:Polygon>',txt)
    ps = re.findall(
        r'<eop:MaskFeature gml:id="CIRRUS[\s\S]*?</eop:MaskFeature>', txt)
    print(len(txt))
    txt.replace(ps[-1], ' ')
    print(len(txt))
    # for i in ps:
    # print(len(txt))
    # txt.replace(i,'')

    # print(txt)
    ps1 = re.findall(
        r'<eop:MaskFeature gml:id="CIRRUS[\s\S]*?</eop:MaskFeature>', txt)
    print(len(ps1))
    write_text('d:/new33333.gml', txt, add=False)

    #p = ogr.CreateGeometryFromGML(ps[522])

    #source_ds  = ogr.Open(gml)

    # outDS.GetRasterBand(1).WriteArray(ccc)
    #outDS = None
    '''
    root = tree.getroot()
    #print(root.tag,root.attrib)
    #node = tree.find()
    print(root)
    for child_of_root in root:
        if child_of_root.tag.endswith('maskMembers'):
            node = child_of_root
            #for n in node:
            p = node.find('.//{http://www.opengis.net/gml/3.2}Polygon')
            #print(p[0].get('{http://www.opengis.net/gml/3.2}id'))
                #print(p[0].items())

    #write_text("d:/t.txt", 'vsws', add=False)
    '''
