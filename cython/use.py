import rpc_trans

if __name__ == '__main__':
    tif_file =  "/mnt/cephfs/rsi/data/test_AT/images/GF2_PMS1_E108.9_N34.2_20181026_L1A0003549596/GF2_PMS1_E108.9_N34.2_20181026_L1A0003549596-MSS1.tiff"
    xs, ys = rpc_trans.cal_boundary_coords(tif_file)
    print(xs, ys)

