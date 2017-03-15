# coding: utf-8
import gdal, osr, ogr, os, pysal
import numpy as np

def main():
    work_dir = r"D:\hcho_change\ROI\COI"
    os.chdir(work_dir)
    os.chdir("aod")

    files = os.listdir(os.getcwd())
    os.chdir("..")
    # name = files[-1]
    for name in files:
        os.chdir("aod")
        print name
        f = gdal.Open(name)
        cols = f.RasterXSize
        rows = f.RasterYSize
        gt = f.GetGeoTransform()
        proj = f.GetProjection()

        array = f.ReadAsArray(0, 0, cols, rows)
        #预处理
        array[np.where(~np.isfinite(array) == True)] = -1
        array[np.where(array < 0)] = np.float("nan")
        #------
        y, x = np.where(np.isfinite(array))

        lats = np.array([gt[3] + gt[5] * i for i in y])
        lons = np.array([gt[0] + gt[1] * i for i in x])
        data = array[y, x]

        coor = np.vstack((lons, lats)).T
        w = pysal.knnW(coor, k =3)
        mr = pysal.Moran_Local(data, w).p_sim
        array[np.where(np.isfinite(array))] = mr
        array[np.where(~np.isfinite(array))] = 0.



        dr = gdal.GetDriverByName("GTiff")
        os.chdir("..")
        os.chdir("out")
        name = name[0 : -4] + "moran.tif"
        ds = dr.Create(name, cols, rows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).WriteArray(array)
        os.chdir("..")
        print name + " has done."

if __name__ == '__main__':
    main()
    print "ok"