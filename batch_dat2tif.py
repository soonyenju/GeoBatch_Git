#coding: utf-8
import os
import numpy as np
import gdal, ogr, os, osr
import shutil

def main():
	arr_size = (720L, 1440L)
	path = r"D:\hcho_change\prods"
	if os.path.isdir(path + "\\out") == True:
		shutil.rmtree(path + "\\out")
	years = os.listdir(path)
	os.mkdir(path + "\\out")
	for i in range(len(years)):
		sub_dir = path + "\\" + years[i]
		data_list = os.listdir(sub_dir)
		arr_mean = np.empty(arr_size)

		for j in range(len(data_list)):
			dat_dir = sub_dir + r"\\" + data_list[j]
			f = open(dat_dir, 'r')
			data = [lines.split() for lines in f.readlines()]
			data =  data[7:]
			array = np.array(data, dtype = np.float64)
			# array = array[::-1, :]
			arr_mean += array/len(data_list)
			dir_name = path + "\\out\\" + years[i] + "_vcd.tif"
			draw_tif(dir_name, arr_mean)
			print j

def draw_tif(dir_name, array):
	cols = 1440
	rows = 720
	originX = -179.88
	originY = -89.88
	pixelWidth = 0.25
	pixelHeight = 0.25

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