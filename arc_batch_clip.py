#coding: utf-8
import arcpy
import os
import numpy as np
def main():
    wrk_dir = r"D:\hcho_change\batch_codes\in"
    msk_dir = r"D:\hcho_change\batch_codes\assist\JJJ.shp"
    # msk_dir = r"D:\hcho_change\Guangdong\GuangdongShp\Guangdong.shp"
    # msk_dir = r"D:\hcho_change\shp\china\china.shp"
    out_dir = r"D:\hcho_change\batch_codes\out"
    start_time = 201501
    clip(wrk_dir, msk_dir, out_dir, start_time, suffix = "_jjj")

def clip(wrk_dir, msk_dir, out_dir, start_time, suffix = "_"):
    mxd = arcpy.mapping.MapDocument("CURRENT")
    arcpy.CheckOutExtension("spatial")
    arcpy.gp.overwriteOutput = 1
    arcpy.env.workspace = wrk_dir + "\\"
    rasters = arcpy.ListRasters("*", "tif")
    # rst_num = len(rasters)
    mask = msk_dir
    for i in range(len(rasters)):
        raster = rasters[i]
        print(raster)
        # out = out_dir + "\\" + str(start_time + i) + suffix + ".tif"
        out = out_dir + "\\" + raster[0 : -4] + suffix + ".tif"
        print raster
        arcpy.gp.ExtractByMask_sa(raster, mask, out)
        print(str(start_time + i) + "  has done")
        dfs = arcpy.mapping.ListDataFrames(mxd)[0]
        lyrs = arcpy.mapping.ListLayers(mxd, "", dfs)
        arcpy.mapping.RemoveLayer(dfs, lyrs[0])
    print("All done")

'''
def create_dir():
    work_dir = r"D:\hcho_change\prods"
    os.chdir(work_dir)
    for parent,dirnames,filenames in os.walk(work_dir):
        if dirnames:
            for i in dirnames:
                os.mkdir(r"D:\hcho_change\china" + "\\" + i)
'''

if __name__ == '__main__':
	main()
	print "ok"