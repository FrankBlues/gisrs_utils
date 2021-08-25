# -*- coding: utf-8 -*-
"""
Module for local reading and writing of data
mostly based on sentinelhub module.

"""

import os
import csv
import json
import struct
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
    from osgeo import gdal
except ImportError:
    print("gdal module missing,install when needed")
    pass

CSV_DELIMITER = ';'


def read_tiff_image(filename):
    """ Read data from TIFF file.

    Args:
        filename (str): Name of TIFF file to be read

    Returns:
        numpy ndarray: Data stored in TIFF file

    """
    return tiff.imread(filename)


def read_raster_gdal(filename):
    """ Read raster data using gdal.

    Args:
        filename (str): Name of raster file to be read.

    Returns:
        tuple: Data array, geotransfom , projection of the data.

    """
    try:
        ds = gdal.Open(filename)
        return (ds.ReadAsArray(), ds.GetGeoTransform(), ds.GetProjection())
    finally:
        del ds


def read_raster_rasterio(filename):
    """ Read raster data using rasterio.

    Args:
        filename (str): Name of raster file to be read.

    Returns:
        tuple: Data array, metadata of the data.

    """
    import rasterio
    with rasterio.open(filename) as src:
        return (src.read(), src.meta)


def read_jp2_image(filename):
    """ Read data from JPEG2000 file.

    Args:
        filename (str): Name of JPEG2000 file to be read.

    Returns:
        numpy ndarray: Data stored in JPEG2000 file

    """
    # Reading with glymur library:
    # return glymur.Jp2k(filename)[:]
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    with open(filename, 'rb') as file:
        bit_depth = get_jp2_bit_depth(file)

    return _fix_jp2_image(image, bit_depth)


def read_image(filename):
    """ Read data from PNG or JPG file.

    Args:
        filename:(str) name of PNG or JPG file to be read.

    Returns:
        numpy ndarray: Data stored in JPG file.

    """
    return cv2.imread(filename, cv2.IMREAD_UNCHANGED)


def read_text(filename):
    """ Read data from text file.

    Args:
        filename (str): Name of text file to be read.

    Returns:
        str: Data stored in text file.

    """
    with open(filename, 'r') as file:
        return file.read()   # file.readline() for reading 1 line


def read_text_lines(filename):
    """ Read data from text file.

    Args:
        filename (str): Name of text file to be read.

    Returns:
        list: Lines in text file.

    """
    with open(filename, 'r') as file:
        return file.readlines()   # file.readline() for reading 1 line


def read_csv(filename, delimiter=CSV_DELIMITER):
    """ Read data from CSV file.

    Args:
        filename (str): Name of CSV file to be read.
        delimiter (str): Type of CSV delimiter. Default is ``,``.

    Returns:
        list: Data stored in CSV file.

    """
    with open(filename, 'r') as file:
        return list(csv.reader(file, delimiter=delimiter))


def read_json(filename):
    """ Read data from JSON file.

    Args:
        filename (str): Name of JSON file to be read.

    Returns:
        dict: Data stored in JSON file.

    """
    with open(filename, 'r') as file:
        return json.load(file)


def read_xml(filename):
    """ Read data from XML or GML file.

    Args:
        filename (str): Name of XML or GML file to be read.

    Returns:
        ElementTree: Data stored in XML file.

    """
    return ElementTree.parse(filename)


def read_numpy(filename):
    """ Read data from numpy file.

    Args:
        filename (str): Name of numpy file to be read.

    Returns:
        array, tuple, dict, etc: Data stored in file.

    """
    return np.load(filename)


def write_tiff_image(filename, image, compress=False):
    """ Write image data to TIFF file.

    Args:
        filename(str) : Name of file to write data to.
        image (numpy ndarray): Image data to write to file.
        compress (bool): Whether or not to compress data. If ``True``,
                    lzma compression is used. Default is ``False``.

    Returns:
        tifffile object.

    """
    if compress:
        # loseless compression, works very well on masks
        return tiff.imsave(filename, image, compress='lzma')
    return tiff.imsave(filename, image)


def write_jp2_image(filename, image):
    """ Write image data to JPEG2000 file

    Args:
        filename (str): Name of JPEG2000 file to write data to.
        image (numpy ndarray): Image data to write to file.

    Returns:
        jp2k object.

    """
    # Writing with glymur library:
    # return glymur.Jp2k(filename, data=image)
    return cv2.imwrite(filename, image)


def write_text(filename, data, add=False):
    """ Write image data to text file.

    Args:
        filename (str): Name of text file to write data to.
        data (numpy ndarray): Image data to write to file.
        add (bool): Whether or not to append to existing file or not.
                Default is ``False``.

    """
    write_type = 'a' if add else 'w'
    with open(filename, write_type) as file:
        file.write(data)


def write_csv(filename, data, delimiter=CSV_DELIMITER):
    """ Write image data to CSV file.

    Args:
        filename (str): Name of CSV file to write data to.
        data (numpy ndarray): Image data to write to CSV file.
        delimiter (str): Delimiter used in CSV file. Default is ``,``.

    """
    with open(filename, 'w') as file:
        csv_writer = csv.writer(file, delimiter=delimiter)
        for line in data:
            csv_writer.writerow(line)


def write_csv_list(filename, data_list, add=False, delimiter=CSV_DELIMITER):
    """ Write list of data to CSV file.

    Args:
        filename (str): Name of CSV file to write data to.
        data_list (list): List of data to write to CSV file.
        delimiter (str): Delimiter used in CSV file. Default is ``,``.

    """
    write_type = 'a' if add else 'w'
    with open(filename, write_type, newline='') as file:
        csv_writer = csv.writer(file, lineterminator='\n')
        csv_writer.writerow(data_list)


def write_json(filename, data):
    """ Write data to JSON file.

    Args:
        filename (str): Name of JSON file to write data to.
        data (list,tuple): Data to write to JSON file.

    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=True)


def write_xml(filename, element_tree):
    """ Write data to XML or GML file.

    Args:
        filename (str): Name of XML or GML file to write data to.
        element_tree (xmlElementTree object): Data as ElementTree object.

    """
    return element_tree.write(filename)
    # this will write declaration tag in first line:
    # return element_tree.write(filename, encoding='utf-8',
    # xml_declaration=True)


def write_numpy(filename, data):
    """ Write data as numpy file.

    Args:
        filename (str): Name of numpy file to write data to.
        data (numpy ndarray): Data to write to numpy file.
    """
    return np.save(filename, data)


def get_jp2_bit_depth(stream):
    """Reads bit encoding depth of jpeg2000 file in binary stream format.

    Args:
        stream (binary): binary stream format.

    Returns:
        bit depth

    Raises:
        ValueError: If image Header Box not found in Jpeg2000 file.

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
    """Because opencv library incorrectly reads jpeg2000 images with 15-bit
    encoding this function corrects the values in image.

    Args:
        image (numpy ndarray): Image read by opencv library.
        bit_depth (int): Bit depth of jp2 image encoding.

    Returns:
        corrected image

    """
    if bit_depth in [8, 16]:
        return image
    if bit_depth == 15:
        return image >> 1
    raise ValueError('Bit depth {} of jp2 image is currently not supported. '
                     'Please raise an issue on package Github page'.format(
                             bit_depth))


def zipDir(outzipfile, zipdir):
    """Zip all files of one directory to zipfile.

    Args:
        outzipfile (str): The output zipfile.
        zipdir (str): Directory to be zipped.

    """
    # ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2 or ZIP_LZMA
    with zipfile.ZipFile(outzipfile, 'w',
                         compression=zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(zipdir):
            for filename in filenames:
                pathfile = os.path.join(dirpath, filename)
                arcname = pathfile[len(os.path.realpath(zipdir)):].strip(
                    os.path.sep)
                z.write(pathfile, arcname)


if __name__ == '__main__':
    pass
