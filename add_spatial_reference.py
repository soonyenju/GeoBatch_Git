#coding: utf-8
import gdal, osr, ogr, os
import numpy as np

def main():
    wrk_dir = r"D:\hcho_change\batch_codes\in"
    out_dir = r"D:\hcho_change\batch_codes\out"
    os.chdir(wrk_dir)
    for parent_dir, child_folders, filenames in os.walk(wrk_dir): pass
    for filename in filenames:
        tif_dir = parent_dir + "\\" + filename
        f = gdal.Open(tif_dir)
        cols = f.RasterXSize
        rows = f.RasterYSize
        data = f.ReadAsArray(0, 0, cols, rows)
        data = data[::-1, :]
        proj = f.GetProjection()
        geo_tran = f.GetGeoTransform()
        print cols, rows, geo_tran
        dir_name = out_dir + "\\" + filename
        draw_tif(dir_name, data)

def draw_tif(dir_name, array):
    cols = 2880
    rows = 1440
    originX = -180
    originY = -90
    pixelWidth = 0.125
    pixelHeight = 0.125

    driver = gdal.GetDriverByName('GTiff')
    newRasterfn = dir_name
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


if __name__ == '__main__':
    main()
    print "ok"