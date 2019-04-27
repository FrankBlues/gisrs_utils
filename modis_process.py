# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:29:31 2018

@author: Administrator
"""
import os
import glob
from subprocess import run, PIPE
import numpy as np
import rasterio
from affine import Affine
import gdal

from mosaic import merge_rio
from io_utils import read_csv


def modis_convert(modis_dir, out_dir):
    """Convert source modis data (hdf format MODIS 13Q1 product) to GTIFF
    format and  projected to UTM 50 zone(epsg:32650).

    Args:
        modis_dir (str): directory contain MXD13Q1 file(both *.hdf and *.xml).
        out_dir (str): directory to store the result file.

    """

    pythonEXE = r'C:\ProgramData\Anaconda3\envs\py333\python.exe'
    modis_convert_script = ('C:/ProgramData/Anaconda3/envs/py333/Scripts'
                            '/modis_convert.py')
    for f in os.listdir(modis_dir):
        outname = f[:23]
        outfile = os.path.join(out_dir, outname)
        infile = os.path.join(modis_dir, f)
        print('Process {}'.format(infile))
        print(outfile)
        if not os.path.exists(outfile + '_250m 16 days NDVI.tif'):

            r = run('{0} {1} -s "( 1 )" -o {2} -e 32650 {3}'.format(pythonEXE,
                    modis_convert_script, outfile, infile),
                    shell=True, stderr=PIPE)
            assert(r.stderr == b'' and r.returncode == 0)
        else:
            print('Result already exits.')


def cal_diff(mfiles_this_year, mfiles_last_year, cal_dir):
    """ Calculate NDVI changes between two year(2018 & 2017),include difference
    and percentage change. Each difference and percentage data is written out
    to a file.

    Args:
        mfiles_this_year (list): MXD13Q1 file list of this year.
        mfiles_last_year (list): MXD13Q1 file list of last year.
        cal_dir (str): directory where the output raster is written to.

    """
    for m in mfiles_this_year:
        mfn = os.path.basename(m)

        datedn = m.split('.')[1]
        m_last_year = m.replace(datedn, datedn.replace('A2018', 'A2017'))

        if m_last_year in mfiles_last_year:
            ds_this_year = rasterio.open(m)
            metas = ds_this_year.meta.copy()

            ds_last_year = rasterio.open(m_last_year)
            # read data
            arr_this_year = ds_this_year.read(1).astype('float32')
            arr_last_year = ds_last_year.read(1).astype('float32')

            arr_this_year[arr_this_year == -3000] = np.nan
            arr_last_year[arr_last_year == -3000] = np.nan
            # calculate difference and change percentage
            diff_arr = (arr_this_year - arr_last_year)*0.0001
            percent_arr = diff_arr/arr_last_year*10000

            diff_arr[np.isnan(diff_arr)] = -999
            percent_arr[np.isnan(percent_arr)] = -999
            percent_arr[np.isinf(percent_arr)] = -999

            # whrite out tiff file
            out_diff_file = os.path.join(cal_dir,
                                         mfn.replace('.tif', '_diff.tif'))
            out_percent_file = os.path.join(cal_dir,
                                            mfn.replace('.tif',
                                                        '_percent.tif'))

            metas.update(nodata=-999, dtype='float32')

            with rasterio.open(out_diff_file, 'w', **metas) as dst:
                dst.write(diff_arr, 1)

            with rasterio.open(out_percent_file, 'w', **metas) as dst:
                dst.write(percent_arr, 1)

            ds_this_year.close()
            ds_last_year.close()


def create_raster_range():
    """ Create raster contain value in certain range.
    This create a raster of the size 1000*1000 and pixcel value range
    from -2000 to 10000, same as the MODIS NDVI value range.

    #TODO make it more general.
    """

    raster = np.zeros((1000, 1000), dtype='int16')

    v = -2000
    for i in range(1000):
        for j in range(1000):
            raster[i, j] = v

            if v == 10000:
                step = -1

            if v == -2000:
                step = 1

            v += step

    with rasterio.open("d:/sd.tif", 'w', driver='GTiff', width=1000,
                       height=1000, count=1, dtype='int16') as dst:
        dst.write(raster, 1)


def mosaic_modis(raster_dir, mosaic_dir):
    """Mosaic modis NDVI data using rasterio.

    Args:
        raster_dir (str): directory containing converted modis ndvi files.
        mosaic_dir (str): output directory.
    """

    # get unique date list
    datelist = []
    for f in os.listdir(raster_dir):
        if f.endswith('.tif'):
            date = f.split('.')[1]
            if date not in datelist:
                datelist.append(date)

    # loop each date,find modis files on that date,then mosaic
    for d in datelist:
        resultlist = glob.glob(os.path.join(out_dir, '*{}*.tif'.format(d)))

        src_files_to_mosaic = [rasterio.open(f) for f in resultlist]
        outMosaic = os.path.join(mosaic_dir,
                                 'MYD13Q1_{}_h28v0506.tif'.format(d))
        merge_rio(src_files_to_mosaic, outMosaic, res=250, nodata=-3000)


def apply_classifiedSymbol(data, symbolfile, out, nodata=255):
    """Convert modis ndvi data to 8 bit raster with classified symbology.

    Args:
        data (str): modis ndvi file
        symbolfile (str): file contains each class symble(RGB value)
        out (out): output symbologized file
        nodata (int): nodata value of the output file
    """
    symbol_list = read_csv(symbolfile, delimiter=',')

    # to numpy ndarray
    color_arr = np.array([l[0:3] for l in symbol_list], dtype='uint8')

    with rasterio.open(data) as src:
        metas = src.meta
        img_arr = src.read(1)
        # mask_arr = src.read_masks(1)
        msk = (img_arr == -3000)
        masked_arr = np.ma.masked_where(msk, img_arr)

        # classified to 100 classes with equel level
        classified = ((masked_arr + 2000)//120.000001).astype('uint8')
        classified = np.ma.filled(classified, nodata)

        metas.update(count=3, nodata=nodata, vdtype='uint8')

        r_arr = np.full((src.height, src.width), nodata, dtype='uint8')
        g_arr = np.full((src.height, src.width), nodata, dtype='uint8')
        b_arr = np.full((src.height, src.width), nodata, dtype='uint8')

        for i in range(100):
            index = (classified == i)
            r_arr[index] = color_arr[i, 0]
            g_arr[index] = color_arr[i, 1]
            b_arr[index] = color_arr[i, 2]

        with rasterio.open(out, 'w', photometric="RGB", **metas) as dst:
            dst.write(r_arr, 1)
            dst.write(g_arr, 2)
            dst.write(b_arr, 3)


def open_eos_hdf_gdal_lst(mxd11c3):
    """Open modis LST(MxD11C3 and the LST_Day_CMG dataset) data using gdal.

    Args:
        mxd11c3 (str): MODIS MOD11C3 or MYD11C3 data file.

    Returns:
        tuple: The data array and gdal geotransform infomation.
    """

    ds = gdal.Open("HDF4_EOS:EOS_GRID:{}:MODIS_MONTHLY_0.05DEG_CMG_LST:"
                   "LST_Day_CMG".format(mxd11c3))
    scale_factor = ds.GetMetadata_Dict()['scale_factor']

    return (ds.ReadAsArray() * float(scale_factor), ds.GetGeoTransform())


def extract_by_coordinate_mxd11c3(MYD_DIR, ul=(123.652796, 46.327),
                                  lr=(128, 42.203257), res=0.05):
    """Extracet modis LST monthly data by coordinate extent.

    Args:
        MYD_DIR (str): Directory holding the myd(mod)11c3 product.
        ul (:tuple: lon,lat): Upper left coordinate.
        lr (:tuple: lon,lat): Lower right coordinate.
        res (float): Resolution in map units.
    """

    for mxd11c3 in glob.glob(os.path.join(MYD_DIR, "*.hdf")):

        lst_arr, trans = open_eos_hdf_gdal_lst(mxd11c3)

        fwd = Affine.from_gdal(trans[0], trans[1], trans[2],
                               trans[3], trans[4], trans[5])
        ul_c, ul_r = ~fwd * ul  # or (123.652796+180)*20, (90-46.700438)*20
        lr_c, lr_r = ~fwd * lr

        ur_lon, ur_lat = fwd * (int(ul_c), int(ul_r))
        trans_old = Affine(res, 0, ur_lon,
                           0, -res, ur_lat)
        width_old = int(lr_c) - int(ul_c) + 1
        height_old = int(lr_r) - int(ul_r) + 1

        kargs = {
                "driver": "GTIFF",
                "count": 1,
                "transform": trans_old,
                "width": width_old,
                "height": height_old,
                "crs": 'epsg:4326',
                "dtype": 'float32',
                }

        src_arr = lst_arr[int(ul_r):int(ul_r) + height_old,
                          int(ul_c):int(ul_c)+width_old].astype('float32')
        with rasterio.open(mxd11c3.replace('.hdf', '.tif'), 'w',
                           **kargs) as dst:
            dst.write(src_arr, 1)


def get_calibrate_bt_params(band_name):
    """Get calibrate parameters from radiance to brightness temperature for
    MODIS L1B emmissive channels(20 - 36,except 26), this function basically
    copied from the satpy module.

    Args:
        band_name (str): Band name of the MODIS emmissive channels in the
        'emmissive_channels' list below.

    Returns:
        tuple: the parameters for calibrating brightness temperature.

    """

    # Planck constant (Joule second)
    h__ = np.float32(6.6260755e-34)

    # Speed of light in vacuum (meters per second)
    c__ = np.float32(2.9979246e+8)

    # Boltzmann constant (Joules per Kelvin)
    k__ = np.float32(1.380658e-23)

    # Derived constants
    c_1 = 2 * h__ * c__ * c__
    c_2 = (h__ * c__) / k__

    # Effective central wavenumber (inverse centimeters)
    cwn = np.array([
        2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3,
        2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3,
        1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2,
        7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
        dtype=np.float32)

    # Temperature correction slope (no units)
    tcs = np.array([
        9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1,
        9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1,
        9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1,
        9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
        dtype=np.float32)

    # Temperature correction intercept (Kelvin)
    tci = np.array([
        4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2,
        7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1,
        1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2,
        1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
        dtype=np.float32)

    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwn = 1. / (cwn * 100)

    # Some versions of the modis files do not contain all the bands.
    emmissive_channels = ["20", "21", "22", "23", "24", "25", "27", "28", "29",
                          "30", "31", "32", "33", "34", "35", "36"]
    global_index = emmissive_channels.index(band_name)

    cwn = cwn[global_index]
    tcs = tcs[global_index]
    tci = tci[global_index]

    return(c_1, c_2, cwn, tcs, tci)


def calibrate_bt_l1b(array, band_name):
    """Calibration for the MODIS l1b emissive channels.

    Args:
        array (numpy ndarray): Radiance of the MODIS l1b emmissive channels.
        band_name (str):  Band name of the MODIS emmissive channels.

    Returns:
        numpy ndarray: The brightness temperature.

    """

    c_1, c_2, cwn, tcs, tci = get_calibrate_bt_params(band_name)
    array = c_2 / (cwn * np.log(c_1 / (1000000 * array * cwn ** 5) + 1))
    array = (array - tci) / tcs
    return array


def bt_to_emissive(array, band_name):
    """Inverse function of the 'calibrate_bt_l1b' to convert the brightness
    temperature back to the radiance."""

    c_1, c_2, cwn, tcs, tci = get_calibrate_bt_params(band_name)

    array = array * tcs + tci
    array = c_1 / (np.e ** (c_2 / array / cwn) - 1) / (1000000 * cwn ** 5)
    return array


def get_attr_from_hdf(hdf, dataset):
    """Get the attributes from the hdf4 file of the given dataset.

    Args:
        hdf (str): Filename of the the NCSA HDF4 file.
        dataset (str): The given dataset in the hdf file.

    Returns:
        dict: The attributes of the given dataset.

    """
    from pyhdf.SD import SD, SDC
    file_handle = SD(hdf, SDC.READ)
    sds_obj = file_handle.select(dataset)
    return sds_obj.attributes()


def calibrate_ref_l1b(arr, attr, band):
    """Calibration for the MODIS reflectance channels.
    """
    index = attr['band_names'].split(',').index(band)
    reflectance_scales = attr['reflectance_scales'][index]
    reflectance_offsets = attr['reflectance_offsets'][index]

    return (arr - reflectance_offsets) * reflectance_scales


def calibrate_rad_l1b(arr, attr, band):
    """Calibration for the MODIS emissive channels.
    """
    index = attr['band_names'].split(',').index(band)
    radiance_scales = attr['radiance_scales'][index]
    radiance_offsets = attr['radiance_offsets'][index]

    return (arr - radiance_offsets) * radiance_scales


def main():
    """Sea fog detection.

    #TODO: 从HDF文件自动提取所需要波段并转换为常用格式

    """

    dmodisl1b = r'H:\sea-fog\data'
    base_name = 'MYD021KM.A2010053.0530.061.2018058131213'
    hdf = os.path.join(dmodisl1b, base_name + '.hdf')
    ref_b01 = os.path.join(dmodisl1b, base_name + '_Ref_B01.tif')
    ref_b02 = os.path.join(dmodisl1b, base_name + '_Ref_B02.tif')
    ref_b18 = os.path.join(dmodisl1b, base_name + '_Ref_B18.tif')
    emi_b20 = os.path.join(dmodisl1b, base_name + '_Emi_B20.tif')
    emi_b31 = os.path.join(dmodisl1b, base_name + '_Emi_B31.tif')

    # Attributes of the 250m band (band 1 and 2)
    attr = get_attr_from_hdf(hdf, 'EV_250_Aggr1km_RefSB')

    def get_raster_arr(r):
        """Read array from a one band raster file."""
        with rasterio.open(r) as src:
            return src.read(1)

    # Calibrate reflectance for band 1 and 2.
    arr_ref_b01 = calibrate_ref_l1b(get_raster_arr(ref_b01), attr, "1")
    arr_ref_b02 = calibrate_ref_l1b(get_raster_arr(ref_b02), attr, "2")

    # Detect sea surface
    sea_surface = np.logical_and(arr_ref_b01 > arr_ref_b02,
                                 np.logical_and(arr_ref_b01 < 0.2,
                                                arr_ref_b02 < 0.2))

    def write_raster(arr, ref_img, out_img, updates):
        """Write array to a raster """
        with rasterio.open(ref_img) as src:
            kargs = src.meta.copy()
            kargs.update(updates)
            with rasterio.open(out_img, 'w', **kargs) as dest:
                dest.write(arr.astype(kargs['dtype']), 1)

    # Write sea surface out
    write_raster(sea_surface, ref_b01, 'D:/ss.tif', {'dtype': 'uint8'})

    # Calibrate radiance for band 20 and 31
    attr = get_attr_from_hdf(hdf, 'EV_1KM_Emissive')
    arr_rad_b31 = calibrate_rad_l1b(get_raster_arr(emi_b31), attr, "31")
    arr_rad_b20 = calibrate_rad_l1b(get_raster_arr(emi_b20), attr, "20")

    # Calibrate brightness temperature of band 31
    bt = calibrate_bt_l1b(arr_rad_b31, "31")

    # Relative high cloud
    cloud_high = np.logical_and(bt < 273,
                                np.logical_and(arr_ref_b01 > 0.55,
                                               arr_ref_b02 > 0.55))

    # Write high cloud out
    write_raster(cloud_high, ref_b01, 'D:/cc.tif', {'dtype': 'uint8'})

    # Calibrate reflectance for band 18
    attr = get_attr_from_hdf(hdf, 'EV_1KM_RefSB')
    arr_ref_b18 = calibrate_ref_l1b(get_raster_arr(ref_b18), attr, "18")

    # Convert the brightness temperature of band 31 to radiance of band 20.
    bt31_to_w20 = bt_to_emissive(bt, "20")

    ifc = (arr_rad_b20 - bt31_to_w20) / arr_ref_b18
    write_raster(ifc, ref_b01, 'D:/ifc.tif', {'dtype': 'float32'})


if __name__ == '__main__':
    modis_dir = r'I:\MODIS\temp'
    out_dir = r'I:\MODIS\h28v0506_conv'

    # modis_convert(modis_dir,out_dir)

    cal_dir = r'I:\MODIS\h28v05_process'
    mosaic_dir = r'I:\MODIS\mosaic'

    maskshp = r'D:\temp\zjs_buffer.shp'
    mask_dir = r'I:\MODIS\mask_'

    mxd11c3 = r"I:\MODIS\MYD11C3\MYD11C3.A2017335.006.2018003155350.hdf"
    MYD_DIR = r"I:\MODIS\MYD11C3"

    myd11a1 = r"I:\MODIS\MYD11A1"

    emissive_b31 = r'D:\WW311.tif'
    bt_b31 = "D:/BT31.tif"
    emi_b20_b31 = "D:/emi_b31_b31.tif"

    dic = {}
    for f in os.listdir(myd11a1):
        full_fname = os.path.join(myd11a1, f)
        date = f.split('.')[1]
        if date not in dic:
            dic[date] = [full_fname]
        else:
            dic[date].append(full_fname)

    # MYD_DIR = r"I:\MODIS\temp"

    # out_arr = np.empty((11,236,297),dtype='float32')
    # i = 0
    # for tif in glob.glob(os.path.join(MYD_DIR,"*_resample_ext.tif")):
    #     print(tif)
    #     with rasterio.open(tif,mask=True) as src:
    #         kargs = src.meta.copy()
    #         arr = src.read(1)
    #         nodata = src.nodata
    #         arr[arr == nodata] = np.nan
    #         out_arr[i,:,:] = arr
    #         i += 1
    # mean = np.nanmean(out_arr,axis = 0)
    # with rasterio.open("D:/mean2.tif",'w',**kargs) as dest:
    #     dest.write(mean,1)
