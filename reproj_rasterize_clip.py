#coding: utf-8
from osgeo import gdal, osr, ogr
import os
import numpy as np

def main():
    gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'NO')
    gdal.SetConfigOption('SHAPE_ENCODING', 'gb2312')
    work_dir = r"D:\hcho_change\china\2007"
    os.chdir(work_dir)

    # reproject()
    raster_fn = r"D:\hcho_change\shp\china.tif"
    shapefile = r"D:\hcho_change\shp\china\china.shp"
    # Rasterize(raster_fn, shapefile)
    path = r"D:\hcho_change\out\2007"
    for i in os.listdir(path):
        file = os.path.join(path, i)
        name = i[-10: -4] + ".tif"
        print file
        clip(raster_fn, file, name)
def reproject(shapefile = r"D:\hcho_change\shp\perl\prd_boundary.shp"):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    inLayer = dataSource.GetLayer()

    inSpatialRef = inLayer.GetSpatialRef()
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(4326)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    geom_type = inLayer.GetGeomType()

    # get the input layer

    # create the output layer
    outputShapefile = "out.shp"
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer("proj_4326",  srs = outSpatialRef, geom_type = geom_type)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    dataSource.Destroy()
    outDataSet.Destroy()

def Rasterize(raster_fn, shapefile):
    pixel_size = 0.25
    NoData_value = -9999

    # raster_fn = r"G:\Pys_HCHO_Workshop\Map\test1.tif"
    # shapefile = r"out.shp"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    inLayer = dataSource.GetLayer()
    x_min, x_max, y_min, y_max = inLayer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    target_ds.SetProjection(outRasterSRS.ExportToWkt())

    im_width = target_ds.RasterXSize # col number
    im_height = target_ds.RasterYSize # row number
    array = np.ones([im_height, im_width])
    target_ds.GetRasterBand(1).WriteArray(array)
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], inLayer, burn_values=[0])

def clip(raster_fn, tiff, name):
    mask = gdal.Open(raster_fn)
    # mask = gdal.Open(r"G:\Pys_HCHO_Workshop\Map\test1.tif")
    mask_geotrans = mask.GetGeoTransform()
    mask_proj = mask.GetProjection()
    mask_data = mask.ReadAsArray(0, 0, mask.RasterXSize, mask.RasterYSize)

    # ds = gdal.Open("test.tif")
    ds = gdal.Open(tiff)
    ds_geotrans = ds.GetGeoTransform()
    ds_proj = ds.GetProjection()
    array = ds.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)

    rows = np.where(mask_data == 0)[0]; cols = np.where(mask_data == 0)[1]
    lons = mask_geotrans[0] + 0.25 * cols
    lats = mask_geotrans[3] - 0.25 * rows

    array[np.where(np.isnan(array))] = 0
    for i in range(rows.shape[0]):
        mask_data[rows[i], cols[i]] = \
            array[np.ceil((lats[i] - ds_geotrans[3])/0.25), np.ceil((lons[i] - ds_geotrans[0])/0.25)]

    cols = mask.RasterXSize
    rows = mask.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    # newRasterfn = "clip.tif"
    newRasterfn = name
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(mask_geotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(mask_data)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

if __name__ == '__main__':
    main()
    print "ok"