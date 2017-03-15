# coding: utf-8
from __future__ import print_function
import arcpy
import os

def main():
    indir = r"D:\hcho_change\batch_codes\in"
    outdir = r"D:\hcho_change\batch_codes\out"
    name = "night_light"
    feats = os.listdir(indir)
    for feat in feats:
        print(feat)
        inFeat = os.path.join(indir, feat)
        outFeat = os.path.join(outdir, feat[0: -4] + name + ".tif")
        install_dir = arcpy.GetInstallInfo()['InstallDir']
        out_coordinate_system = os.path.join(install_dir, r"Coordinate Systems/Projected Coordinate Systems/UTM/NAD 1983/NAD 1983 UTM Zone 11N.prj")
        # out_coordinate_system = os.path.join(install_dir, r"Coordinate Systems/Geographic Coordinates Systems/World/WGS 1984.proj")
        # arcpy.Project_management(inFeat, outFeat, out_coordinate_system) # for vector files
        targetSR = arcpy.SpatialReference(4326)
        arcpy.ProjectRaster_management(inFeat, outFeat, targetSR)
        print(feat + " is done.")


if __name__ == '__main__':
    main()
    print("OK, all done.")