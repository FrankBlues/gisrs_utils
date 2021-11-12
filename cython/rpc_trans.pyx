
cimport cython
include "gdal.pxi"


def cal_boundary_coords(tif_file):
    """根据RPC计算影像四角坐标"""
    GDALAllRegister()
    # file_name = tif_file.encode('utf-8')
    # cdef const char *file_name = unicode
    ds_h = GDALOpen(tif_file.encode('utf-8'), <GDALAccess>0)
    # 宽高
    cdef double width
    cdef double height
    width = <double>GDALGetRasterXSize(ds_h)
    height = <double>GDALGetRasterYSize(ds_h)

    # RPC信息
    papszRPC = GDALGetMetadata(ds_h, "RPC")

    cdef GDALRPCInfo oInfo
    GDALExtractRPCInfo(papszRPC, &oInfo)

    # DEM
    # cdef char** papszTransOption = NULL
    # papszTransOption = CSLSetNameValue(papszTransOption, "RPC_DEM", "/mnt/cephfs/rsi/data/test_AT/dem/DEM_YANTA.tif")

    # RPC Transformer
    pRPCTransform = GDALCreateRPCTransformer(&oInfo, 0, 0, NULL)

    # 需要转换的4个角点坐标
    cdef double dX[4]
    dX = [0.0, 0.0, width, width]
    cdef double dY[4] 
    dY = [0, height, height, 0]
    cdef double dZ[4]
    dZ = [0]*4
    cdef int nSuccess[4]
    nSuccess = [0] * 4
    # 转换
    GDALRPCTransform(pRPCTransform, 0, 4, dX, dY, dZ, nSuccess);

    GDALClose(ds_h);
    GDALDestroyRPCTransformer(pRPCTransform);

    return dX, dY



