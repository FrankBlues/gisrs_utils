# GDAL API definitions.

cdef extern from "cpl_string.h" nogil:
    char **CSLSetNameValue(char **list, char *name, char *val)


cdef extern from "gdal.h" nogil:
    ctypedef void * GDALMajorObjectH
    ctypedef void * GDALDatasetH
    ctypedef void * GDALDataset

    ctypedef enum GDALDataType:
        GDT_Unknown
        GDT_Byte
        GDT_UInt16
        GDT_Int16
        GDT_UInt32
        GDT_Int32
        GDT_Float32
        GDT_Float64
        GDT_CInt16
        GDT_CInt32
        GDT_CFloat32
        GDT_CFloat64
        GDT_TypeCount

    ctypedef enum GDALAccess:
        GA_ReadOnly
        GA_Update

    ctypedef struct GDALRPCInfo:
        double dfLINE_OFF
        double dfSAMP_OFF
        double dfLAT_OFF
        double dfLONG_OFF
        double dfHEIGHT_OFF

        double dfLINE_SCALE
        double dfSAMP_SCALE
        double dfLAT_SCALE
        double dfLONG_SCALE
        double dfHEIGHT_SCALE

        double adfLINE_NUM_COEFF[20]
        double adfLINE_DEN_COEFF[20]
        double adfSAMP_NUM_COEFF[20]
        double adfSAMP_DEN_COEFF[20]

        double dfMIN_LONG
        double dfMIN_LAT
        double dfMAX_LONG
        double dfMAX_LAT
    
    int GDALExtractRPCInfo(char **papszMD, GDALRPCInfo * )
    void GDALAllRegister()
    GDALDatasetH GDALOpen(const char *filename, GDALAccess access) # except -1
    void GDALClose(GDALDatasetH hds)
    int GDALGetRasterXSize(GDALDatasetH hds)
    int GDALGetRasterYSize(GDALDatasetH hds)
    char** GDALGetMetadata(GDALMajorObjectH obj, const char *pszDomain)


cdef extern from "gdal_alg.h" nogil:
    void *GDALCreateRPCTransformer( GDALRPCInfo *psRPC, int bReversed,
                          double dfPixErrThreshold,
                          char **papszOptions )
    void GDALDestroyRPCTransformer( void *pTransformArg )
    int GDALRPCTransform(void *pTransformArg, int bDstToSrc,
                         int nPointCount, double *x, double *y, double *z,
                         int *panSuccess )