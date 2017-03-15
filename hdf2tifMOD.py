# coding: utf-8
import gdal, osr, ogr, os
from pyhdf.HDF import *
from pyhdf.V   import *
from pyhdf.VS  import *
from pyhdf.SD  import *
import numpy as np
import pandas as pd

def main():
    indir = r"D:\hcho_change\Guangdong\MOD\in"
    outdir = r"D:\hcho_change\Guangdong\MOD\out"
    files = os.listdir(indir)
    inname = files[0]
    for file in files:
        inname = file
        outname = file[9: 16]
        mon = str(int(outname[4::]) / 30 + 1)
        if len(mon) == 2:
            outname = outname[0: 4] + mon
        else:
            outname = outname[0: 4] + "0" + mon
        proc(indir, outdir, inname, outname)
        print outname

def proc(indir, outdir, inname, outname):
    path = indir + "/" + inname
    hdf = HDF(path)
    sd = SD(path)
    vs = hdf.vstart()
    v  = hdf.vgstart()
    mod_vg = v.attach("MOD_Grid_monthly_CMG_VI")
    vg_members = mod_vg.tagrefs()
    # print vg_members
    mod_vg = v.attach("MOD_Grid_monthly_CMG_VI")
    tag, ref = mod_vg.tagrefs()[0]
    # print tag, ref
    vg0 = v.attach(ref)
    # print vg0._name
    tagrefs = vg0.tagrefs()
    # print tagrefs
    for tag, ref in tagrefs:
        if tag == HC.DFTAG_NDG:
            sds = sd.select(sd.reftoindex(ref))
            name = sds.info()[0]
            # print name
            if name == "CMG 0.05 Deg Monthly NDVI":
                sd = SD(path)
                sds = sd.select(sd.reftoindex(ref))
                ndvi = np.float64(sds.get())
                sds.endaccess()
            elif name == "CMG 0.05 Deg Monthly EVI":
                sd = SD(path)
                sds = sd.select(sd.reftoindex(ref))
                evi = np.float64(sds.get())
                sds.endaccess()
    sd.end()
    v.end()

    data = ndvi
    name = outdir + "/" + outname + ".tif"
    cols = 7200
    rows = 3600
    originX = -180.0
    originY = 90.0
    pixelWidth = 0.05
    pixelHeight = -0.05

    driver = gdal.GetDriverByName('GTiff')
    newRasterfn = name
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

if __name__ == '__main__':
    main()
    print "ok"