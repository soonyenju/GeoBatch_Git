# coding: utf-8
import gdal, osr, ogr, os
import numpy as np

def main():
    work_dir = r"D:\hcho_change\ROI\COI"
    os.chdir(work_dir)
    os.chdir("hcho")

    files = os.listdir(os.getcwd())
    name = files[-1]
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
    print gt
    lats = np.array([gt[3] + gt[5] * i for i in y])
    lons = np.array([gt[0] + gt[1] * i for i in x])
    data = array[y, x]

    data = np.vstack((lons, lats, data)).T

    filename = r"D:\hcho_change\ROI\COI\hcho2013.shp"
    dr = ogr.GetDriverByName("ESRI Shapefile")
    ds = dr.CreateDataSource(filename)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    geomtype = ogr.wkbPoint

    lyr = ds.CreateLayer("workshop", srs = sr, geom_type = geomtype)
    field = ogr.FieldDefn("data", ogr.OFTReal)
    field.SetWidth(100)
    lyr.CreateField(field)

    for i in range(data.shape[0]):
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetField("data", data[i, 2])

        wkt = "POINT(%f %f)" % (data[i, 0], data[i, 1])

        point = ogr.CreateGeometryFromWkt(wkt)
        feat.SetGeometry(point)
        lyr.CreateFeature(feat)
        feat.Destroy()

if __name__ == '__main__':
    main()
    print "ok"