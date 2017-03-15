# coding: utf-8
import gdal, osr, ogr, os, csv
import numpy as np
import pandas as pd

def main():
    #------------------------------------------
    wrk_dir = r"D:\hcho_change\batch_codes\in"
    out_dir = r"D:\hcho_change\batch_codes\out"
    start_year = 2005
    #------------------------------------------
    count = 0
    os.chdir(wrk_dir)
    files = os.listdir(os.getcwd())
    for file in files:
        f = gdal.Open(file)
        cols = f.RasterXSize
        rows = f.RasterYSize
        array = f.ReadAsArray(0, 0, cols, rows)
        array[np.where(array < 0)] = 0
        values = np.unique(array)
        info = []
        for v in values:
            n = np.where(v == array)[0].shape[0]
            info.append((v, n))
            print (v, n)
        df = pd.DataFrame(info)
        name = str(start_year + count)
        df.to_csv(out_dir + "\\" + name + ".csv")
        count = count + 1
        print name + " has done!"



if __name__ == '__main__':
    main()
    print "All done."