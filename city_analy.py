# coding: utf-8
import os, gdal, osr, ogr, cPickle as pickle
import numpy as np
import pandas as pd

def main():
    wrk_dir = r"D:\hcho_change\Whole_China"
    # save_info(wrk_dir)
    # cityinfo_save(wrk_dir)
    # draw_annual_cities(wrk_dir)
    # draw_ssns_cities(wrk_dir)

def call_correlate():
    # 仅调用correlate()函数，没别的用
    file = open(r"D:\hcho_change\Whole_China\out\sate",'rb')
    sate = pickle.load(file)
    file.close()
    ssn = cal_seasons()

    idx = np.where(ssn != "wtr")
    hcho = sate["hcho"]; hgt = sate["hgeo"]; hkeys = np.sort(hcho.keys())
    aod = sate["aod"]; agt = sate["ageo"]; akeys = np.sort(aod.keys())
    ndvi = sate["ndvi"]; dgt = sate["ndvigeo"]; dkeys = np.sort(ndvi.keys())
    hkeys = hkeys[idx]; akeys = akeys[idx]; dkeys = dkeys[idx]
    # correlate(hcho, hgt, hkeys, aod, agt, akeys, name = "ha_p_fal.tif")
    correlate(hcho, hgt, hkeys, ndvi, dgt, dkeys, name = "hn_p_wtr.tif")

def correlate(data1, gt1, keys1, data2, gt2, keys2, name = "corr_pearson.tif", mode = 'p'):
    shape1 = data1[keys1[0]].shape; shape2 = data2[keys2[0]].shape
    array = np.zeros(shape1)
    lons1 = np.array([gt1[0] + gt1[1] * i for i in range(shape1[1])])
    lats1 = np.array([gt1[3] + gt1[5] * i for i in range(shape1[0])])
    lons2 = np.array([gt2[0] + gt2[1] * i for i in range(shape2[1])])
    lats2 = np.array([gt2[3] + gt2[5] * i for i in range(shape2[0])])

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            lat = lats1[i]; lon = lons1[j]
            row = np.abs(lats2 - lat).argmin()
            col = np.abs(lons2 - lon).argmin()

            vec1 = np.array([data1[key][i, j] for key in keys1])
            # vec2 = np.array([data2[key][row, col] for key in keys2])
            vec2 = np.array([w_mean(data2[key], row, col) for key in keys2])
            if True in np.isnan(vec1) or True in np.isnan(vec2):
                idx1 = np.where(np.isfinite(vec1) == True)[0]
                idx2 = np.where(np.isfinite(vec2) == True)[0]
                idx = np.intersect1d(idx1, idx2)
                vec1 = vec1[idx]; vec2 = vec2[idx]
            if mode == 'p':
                array[i, j] = pearson(vec1, vec2)
            elif mode == 's':
                array[i, j] = spearman(vec1, vec2)
    # 后处理
    array[np.where(np.isfinite(array) == False)] = 0
    array[np.where(array > 1)] = 1; array[np.where(array < -1)] = -1
    draw_tif(name, array, array.shape, gt1)

def w_mean(data, row, col, ws = 3): # 改窗口大小以后再说，现在只能3x3
    vals = data[row - 2: row + 3, col - 2 : col + 3]
    vals = vals[np.where(np.isnan(vals) == False)]
    if vals.size:
        val = vals.mean()
        return val
    else:
        return 0

def spearman(vec1, vec2):
    vec1 = np.array(vec1); vec2 = np.array(vec2)
    vec1_ = vec1.copy(); vec1_ = np.sort(vec1_)[::-1]
    vec2_ = vec2.copy(); vec2_ = np.sort(vec2_)[::-1]

    if np.isfinite(vec1).all() == False or np.isfinite(vec2).all() == False:
        s = 0
    else:
        trace_1 = np.array([np.where(vec1_ == vec1[i])[0][0] for i in range(vec1.shape[0])])
        trace_2 = np.array([np.where(vec2_ == vec2[i])[0][0] for i in range(vec2.shape[0])])
        dif_t = trace_1 - trace_2
        n = dif_t.shape[0]
        s = 1 - 6 * np.float(np.sum((dif_t)**2)) / np.float((n * (n**2 - 1)))

    return s

def pearson(vec1, vec2):
    lens = len(vec1)
    if lens != 0 and np.any((vec1 + vec2) != 0):
        vec1 = (vec1 - vec1.min()) / (vec1.max() - vec1.min())
        vec2 = (vec2 - vec2.min()) / (vec2.max() - vec2.min())
        s_cross = sum([vec1[i] * vec2[i] for i in range(lens)])
        s_vec1 = sum(vec1); s_vec2 = sum(vec2)
        s_vec1sq = sum([vec1[i] * vec1[i] for i in range(lens)])
        s_vec2sq = sum([vec2[i] * vec2[i] for i in range(lens)])

        p_numerator = s_cross - (s_vec1 * s_vec2) / lens
        p_denominator_l = np.sqrt(np.abs(s_vec1sq - ((s_vec1)**2/lens)))
        p_denominator_r = np.sqrt(np.abs(s_vec2sq - ((s_vec2)**2/lens)))

        p = p_numerator / (p_denominator_l * p_denominator_r)
    else:
        p = 0
    return p

def draw_tif(name, array, shape, gt):
    import gdal, osr
    cols = shape[1]
    rows = shape[0]
    originX = gt[0]
    originY = gt[3]
    pixelWidth = gt[1]
    pixelHeight = gt[5]

    driver = gdal.GetDriverByName('GTiff')
    newRasterfn = name
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def draw_ssns_cities(wrk_dir):
    """
    画相关图
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # wrk_dir = r"D:\hcho_change\Whole_China"
    hcho = pd.read_excel(os.path.join(wrk_dir, r"out\hcho_record.xlsx"))
    aod = pd.read_excel(os.path.join(wrk_dir, r"out\aod_record.xlsx"))
    nlight = pd.read_excel(os.path.join(wrk_dir, r"out\nlight_record.xlsx"))
    no2 = pd.read_excel(os.path.join(wrk_dir, r"out\no2_record.xlsx"))
    ndvi = pd.read_excel(os.path.join(wrk_dir, r"out\ndvi_record.xlsx"))
    o3 = pd.read_excel(os.path.join(wrk_dir, r"out\o3_record.xlsx"))

    yrs = np.array([2005 + i for i in range(11)])
    yr_mons = hcho.index
    cities = hcho.columns

    for site in range(cities.shape[0]):
    # site = 0
        h = hcho[cities[site]]
        a = aod[cities[site]]
        n = no2[cities[site]]
        d = ndvi[cities[site]]
        o = o3[cities[site]]
        valsH = h.values
        valsA = a.values
        valsN = n.values
        valsD = d.values
        valsO = o.values
        valsHNorm = (valsH - valsH.min()) / (valsH.max() - valsH.min())
        valsANorm = (valsA - valsA.min()) / (valsA.max() - valsA.min())
        valsNNorm = (valsN - valsN.min()) / (valsN.max() - valsN.min())
        valsDNorm = (valsD - valsD.min()) / (valsD.max() - valsD.min())
        valsONorm = (valsO - valsO.min()) / (valsO.max() - valsO.min())
        years = h.index; years = np.array([str(i)[0: -2] for i in years])
        months = h.index; months = np.array([str(i)[-2::] for i in months])
        seasons = np.empty(months.shape, dtype = "S32")

        seasons[0::12] = "wtr"; seasons[1::12] = "wtr"; seasons[11::12] = "wtr"
        seasons[2::12] = "spr"; seasons[3::12] = "spr"; seasons[4::12] = "spr"
        seasons[5::12] = "smr"; seasons[6::12] = "smr"; seasons[7::12] = "smr"
        seasons[8::12] = "fal"; seasons[9::12] = "fal"; seasons[10::12] = "fal"

        df = pd.DataFrame([yr_mons, years, months, seasons, valsH, valsHNorm, valsA, valsANorm, valsN, valsNNorm, valsD, valsDNorm, valsO, valsONorm],
            index = ["timepoint", "years", "months", "seasons", "hcho", "normed_hcho", "aod", "normed_aod", "no2", "normed_no2", "ndvi", "normed_ndvi", "o3", "normed_o3"]).T
        # print help(df["hcho"][1::].diff)
        df_diff = df["hcho"].diff(); df_diff[0] = 0
        df["diff_hcho"] = df_diff
        df_diff = df["aod"].diff(); df_diff[0] = 0
        df["diff_aod"] = df_diff
        df_diff = df["no2"].diff(); df_diff[0] = 0
        df["diff_no2"] = df_diff
        df_diff = df["ndvi"].diff(); df_diff[0] = 0
        df["diff_ndvi"] = df_diff
        df_diff = df["o3"].diff(); df_diff[0] = 0
        df["diff_o3"] = df_diff

        # df_ = df[["normed_aod", "normed_hcho", "normed_ndvi", "normed_no2", "normed_o3"]].astype(np.float)
        df_ = df[["normed_aod", "normed_hcho", "normed_ndvi"]].astype(np.float)
        # df_["seasons"] = df["seasons"]
        try:
            # g = sns.PairGrid(df_, hue="seasons")
            # g.map_diag(plt.hist)
            # g.map_upper(plt.scatter)
            # g.add_legend();
            # g = sns.pairplot(df_, kind="reg", diag_kind = "kde")
            # g.title = cities[site] + " analysis"
            # g = sns.pairplot(df_, kind="reg", hue = "seasons", diag_kind = "kde")
            g = sns.pairplot(df_, kind="reg", diag_kind = "kde")
            g.savefig(os.path.join(wrk_dir, "out\\" + cities[site] + ".jpg"))
            plt.close("all")
        except Exception, e:
            print Exception,":",e
            continue
        print cities[site] + " is done."

def draw_annual_cities(wrk_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # wrk_dir = r"D:\hcho_change\Whole_China"
    hcho = pd.read_excel(os.path.join(wrk_dir, r"out\hcho_record.xlsx"))
    aod = pd.read_excel(os.path.join(wrk_dir, r"out\aod_record.xlsx"))
    nlight = pd.read_excel(os.path.join(wrk_dir, r"out\nlight_record.xlsx"))

    yrs = np.array([2005 + i for i in range(11)])
    yr_mons = hcho.index
    cities = hcho.columns

    for site in range(cities.shape[0]):
        # site = 0
        h = hcho[cities[site]]
        a = aod[cities[site]]
        valsH = h.values
        valsA = a.values
        valsHNorm = (valsH - valsH.min()) / (valsH.max() - valsH.min())
        valsANorm = (valsA - valsA.min()) / (valsA.max() - valsA.min())
        years = h.index; years = np.array([str(i)[0: -2] for i in years])
        months = h.index; months = np.array([str(i)[-2::] for i in months])
        seasons = np.empty(months.shape, dtype = "S32")

        seasons[0::12] = "wtr"; seasons[1::12] = "wtr"; seasons[11::12] = "wtr"
        seasons[2::12] = "spr"; seasons[3::12] = "spr"; seasons[4::12] = "spr"
        seasons[5::12] = "smr"; seasons[6::12] = "smr"; seasons[7::12] = "smr"
        seasons[8::12] = "fal"; seasons[9::12] = "fal"; seasons[10::12] = "fal"

        df = pd.DataFrame([yr_mons, years, months, seasons, valsH, valsHNorm, valsA, valsANorm],
            index = ["timepoint", "years", "months", "seasons", "hcho", "normed_hcho", "aod", "normed_aod"]).T
        # print help(df["hcho"][1::].diff)
        df_diff = df["hcho"].diff(); df_diff[0] = 0
        df["diff_hcho"] = df_diff
        df_diff = df["aod"].diff(); df_diff[0] = 0
        df["diff_aod"] = df_diff

        # print df
        try:
            fig = plt.figure(figsize = (24,12), dpi = 800)
            # fig = plt.figure()
            fig.suptitle(cities[site] + " analysis", fontsize = 32)
            sns.set_style("darkgrid", {'xtick.direction': u'in'})
            #===============================================================
            plt.subplot(421)
            ax = plt.gca()

            sns.barplot(x = "timepoint", y = "hcho", data = df);
            yerdata = cal_years(df["hcho"])

            plt.plot(np.linspace(5, 127, num = 11), yerdata, "^-", lw = 3, markersize=12, label = "year average")
            plt.legend(loc = 2)
            plt.title("2005-2015 HCHO concentration trend", {"fontsize": "large"})

            ax.set_xticks([5 + i * 12 for i in range(11)])
            ax.set_xticklabels(("2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"))
            plt.xticks(rotation = 15)

            #===============================================================
            plt.subplot(422)
            sns.barplot(x = "years", y = "hcho", hue = "seasons", data = df);
            plt.title("2005-2015 HCHO seasonal distribution", {"fontsize": "large"})

            #===============================================================
            plt.subplot(423)
            ax = plt.gca()
            sns.barplot(x = "timepoint", y = "aod", data = df);
            yerdata = cal_years(df["aod"])

            plt.plot(np.linspace(5, 127, num = 11), yerdata, "v-", lw = 3, markersize=12, label = "year average")
            plt.legend(loc = 2)
            plt.title("2005-2015 AOD concentration trend", {"fontsize": "large"})

            ax.set_xticks([5 + i * 12 for i in range(11)])
            ax.set_xticklabels(("2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"))
            plt.xticks(rotation = 15)

            #===============================================================
            plt.subplot(424)
            sns.barplot(x = "years", y = "aod", hue = "seasons", data = df);
            # help(sns.barplot)
            plt.title("2005-2015 AOD concentration trend", {"fontsize": "large"})

            #===============================================================

            #===============================================================
            plt.subplot(425)
            ax = plt.gca()
            sns.pointplot(x = "timepoint", y = "diff_hcho", data = df);
            yerdata = cal_years(df["diff_hcho"])

            plt.plot(np.linspace(5, 127, num = 11), yerdata, "rv-", lw = 3, markersize=12, label = "year average")
            plt.legend(loc = 2)
            ax.set_xticks([5 + i * 12 for i in range(11)])
            ax.set_xticklabels(("2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"))
            plt.xticks(rotation = 15)
            plt.title("Diff_HCHO", {"fontsize": "large"})
            #===============================================================

            #===============================================================
            plt.subplot(426)
            sns.pointplot(x = nlight.index, y = cities[site], data = nlight);
            plt.title("Economical trend(nlight light)", {"fontsize": "large"})
            #===============================================================

            #===============================================================
            plt.subplot(427)
            ax = plt.gca()
            sns.pointplot(x = "timepoint", y = "diff_aod", data = df);
            yerdata = cal_years(df["diff_aod"])

            plt.plot(np.linspace(5, 127, num = 11), yerdata, "rv-", lw = 3, markersize=12, label = "year average")
            plt.legend(loc = 2)
            ax.set_xticks([5 + i * 12 for i in range(11)])
            ax.set_xticklabels(("2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"))
            plt.xticks(rotation = 15)
            plt.title("Diff_AOD", {"fontsize": "large"})
            #===============================================================

            #===============================================================
            plt.subplot(428)
            sns.boxplot(x = "normed_hcho", y = "seasons", data = df);
            sns.stripplot(x = "normed_hcho", y = "seasons", data = df, jitter=True);
            plt.title("HCHO seasonal distribution", {"fontsize": "large"})
            #===============================================================

            #===============================================================
            plt.subplots_adjust(hspace = 0.5)
            # plt.show()
            fig.savefig(os.path.join(wrk_dir, "out\\" + cities[site] + ".jpg"))
            plt.close("all")
            #===============================================================

            """
            # 画图显示标题等字体
            params = {'legend.fontsize': 'x-large',
                      'figure.figsize': (15, 5),
                     'axes.labelsize': 'x-large',
                     'axes.titlesize':'x-large',
                     'xtick.labelsize':'x-large',
                     'ytick.labelsize':'x-large'}
            pylab.rcParams.update(params)
            """
        except Exception, e:
            print Exception,":",e
            continue
        print cities[site] + " is done."

def combine_ssn(spr, smr, fal, wtr, start_year):
    yrs = np.array([start_year + i for i in range(spr.shape[0])])
    spr_ = np.repeat("spr", spr.shape[0], axis = 0)
    smr_ = np.repeat("smr", smr.shape[0], axis = 0)
    fal_ = np.repeat("fal", fal.shape[0], axis = 0)
    wtr_ = np.repeat("wtr", wtr.shape[0], axis = 0)
    yrs = np.hstack([yrs, yrs, yrs, yrs])
    ssn_ = np.hstack([spr_, smr_, fal_, wtr_])
    ssn = np.hstack([spr, smr, fal, wtr])
    return pd.DataFrame([yrs, ssn, ssn_], index = ["years", "data", "season"]).T

def cal_years(vals):
    yr_data = np.array([vals[i::12].mean() for i in range(len(vals) / 12)])
    return yr_data

def cal_seasons(vals = 0, mode = 2, mons = 132):
    if mode == 1:
        spr = (vals[2::12] + vals[3::12] + vals[4::12]) / 3
        smr = (vals[5::12] + vals[6::12] + vals[7::12]) / 3
        fal = (vals[8::12] + vals[9::12] + vals[10::12]) / 3
        wtr = (vals[12::12] + vals[13::12] + vals[11: -1 :12]) / 3
        fir = (vals[0] + vals[1]) / 2
        wtr = np.r_[fir, wtr]
        return spr, smr, fal, wtr

    elif mode == 2:
        seasons = np.empty(mons, dtype = "S32")
        seasons[0::12] = "wtr"; seasons[1::12] = "wtr"; seasons[11::12] = "wtr"
        seasons[2::12] = "spr"; seasons[3::12] = "spr"; seasons[4::12] = "spr"
        seasons[5::12] = "smr"; seasons[6::12] = "smr"; seasons[7::12] = "smr"
        seasons[8::12] = "fal"; seasons[9::12] = "fal"; seasons[10::12] = "fal"
        return seasons

def cityinfo_save(wrk_dir):
    """
    保存各个城市的hcho，aod和nlight的十年月均数据，依赖于rec_city()
    """
    file = open(r"D:\hcho_change\Whole_China\out\sate",'rb')
    sate = pickle.load(file)
    file.close()

    cityinfo = pd.read_csv(os.path.join(wrk_dir, r"out\city.csv"))
    cis = cityinfo.drop(cityinfo.columns[0], axis = 1)

    geo = sate["hgeo"]
    rec_city(wrk_dir, sate, "hcho", geo, cis)
    geo = sate["ageo"]
    rec_city(wrk_dir, sate, "aod", geo, cis)
    geo = sate["ngeo"]
    rec_city(wrk_dir, sate, "nlight", geo, cis)
    geo = sate["no2geo"]
    rec_city(wrk_dir, sate, "no2", geo, cis)
    geo = sate["ndvigeo"]
    rec_city(wrk_dir, sate, "ndvi", geo, cis)
    geo = sate["o3geo"]
    rec_city(wrk_dir, sate, "o3", geo, cis)

def rec_city(wrk_dir, sate, pol_name, geo, cis):
    """
    对每个点按经纬度查询保存
    """
    dict_pol = sate[pol_name]
    dates = np.sort(dict_pol.keys())
    record = np.empty([cis["city"].shape[0], dates.shape[0]])
    for i in cis.index:
        city, lon, lat = cis.iloc[i, :]
        for j in range(dates.shape[0]):
            date = dates[j]
            data = dict_pol[date]
            col, row = coor2pos(lon, lat, geo)
            vals = data[row - 2: row + 3, col - 2 : col + 3]
            vals = vals[np.where(np.isnan(vals) == False)]
            if vals.size:
                val = vals.mean()
                record[i, j] = val
            else:
                record[i, j] = 0
    df = pd.DataFrame(record.T, columns = cis["city"], index = dates)
    df.to_csv(os.path.join(wrk_dir, r"out\\" + pol_name + "_record.csv")) # 下次改成Excel，更好读
    print pol_name + " is done."

def coor2pos(lon, lat, geo):
    """
    通过经纬度计算下标
    """
    # print geo[0] + geo[1]*cols
    # print geo[3] + geo[5]*rows
    col = np.round((lon - geo[0]) / geo[1])
    row = np.round((lat - geo[3]) / geo[5])
    return col, row

def save_info(wrk_dir):
    """
    hcho、aod、nlight等十年数据保存到一个字典
    """
    cityinfo = get_cityinfo(wrk_dir)
    cityinfo.to_csv(os.path.join(wrk_dir, r"out\city.csv"))
    aod, ageo = get_info(wrk_dir, "aod")
    hcho, hgeo = get_info(wrk_dir, "hcho")
    nlight, ngeo = get_info(wrk_dir, "nlight")
    no2, no2geo = get_info(wrk_dir, "no2")
    ndvi, ndvigeo = get_info(wrk_dir, "ndvi")
    o3, o3geo = get_info(wrk_dir, "o3")

    sate = {"aod": aod, "hcho": hcho, "nlight": nlight, "no2": no2,
             "ndvi": ndvi, "o3": o3,
             "ageo": ageo, "hgeo": hgeo, "ngeo": ngeo, "no2geo": no2geo,
             "ndvigeo": ndvigeo,"o3geo": o3geo}
    # np.save(os.path.join(wrk_dir, r"out\sateData.npy"), sate)
    print "start"
    file = open(r"D:\hcho_change\Whole_China\out\sate",'wb')
    pickle.dump(sate, file, -1)
    file.close()

def get_info(wrk_dir, type):
    dir_ = os.path.join(wrk_dir, type)
    dirs = os.listdir(dir_)
    data = {}
    for dir in dirs:
        dir = os.path.join(dir_, dir)
        name = dir.split("\\")[4][0: 6]
        ds = gdal.Open(dir)
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        # bands = ds.RasterCount
        geotrans = ds.GetGeoTransform()
        proj = ds.GetProjection()
        array = ds.ReadAsArray(0, 0, cols, rows)
        array[np.where(array < 0)] = 0
        array[np.where(array > 60000)] = 0
        data[name] = array
    return data, geotrans

def get_cityinfo(wrk_dir):
    cityDir = os.path.join(wrk_dir, r"ChinaResidence\ChinaResidents.shp")
    ds = ogr.Open(cityDir, False)
    lyr = ds.GetLayer(0)
    spaRef = lyr.GetSpatialRef()
    lyrDef = lyr.GetLayerDefn()
    fieldList = []
    for i in range(lyrDef.GetFieldCount()):
        fieldDef = lyrDef.GetFieldDefn(i)
        fieldDict = {"name": fieldDef.GetName(), "type": fieldDef.GetType(),
                        "width": fieldDef.GetWidth(), "decimal": fieldDef.GetPrecision()}
        fieldList.append(fieldDict)
    geoList, recList = [], []
    feat = lyr.GetNextFeature()
    while feat is not None:
        geo = feat.GetGeometryRef()
        geoList.append(geo.ExportToWkt())
        # print geo.ExportToWkt()
        rec = {}
        for fd in fieldList:
            rec[fd["name"]] = feat.GetField(fd["name"])
        recList.append(rec)
        feat = lyr.GetNextFeature()

    names, lons, lats = [], [], []
    for rec in recList:
        names.append(rec["PINYIN"])
        lons.append(rec["POINT_X"])
        lats.append(rec["POINT_Y"])
    df = pd.DataFrame([names, lons, lats], index = ["city", "lon", "lat"]).T
    # df.to_csv(os.path.join(wrk_dir, "city.csv"))
    return df

"""
#之前的辅助代码
def draw_almost_done():
    import seaborn as sns
    import matplotlib.pyplot as plt
    wrk_dir = r"D:\hcho_change\Whole_China"
    hcho = pd.read_csv(os.path.join(wrk_dir, r"out\hcho_record.csv"))
    aod = pd.read_csv(os.path.join(wrk_dir, r"out\aod_record.csv"))
    nlight = pd.read_csv(os.path.join(wrk_dir, r"out\nlight_record.csv"))

    years = np.array([2005 + i for i in range(11)])

    h_cname = hcho.columns
    hcho = hcho.set_index(h_cname[0])
    hcho_yrs = cal_years(hcho[h_cname[1]])
    hcho_yrs = (hcho_yrs - hcho_yrs.min()) / (hcho_yrs.max() - hcho_yrs.min())

    a_cname = aod.columns
    aod = aod.set_index(a_cname[0])
    aod_yrs = cal_years(aod[a_cname[1]])
    aod_yrs = (aod_yrs - aod_yrs.min()) / (aod_yrs.max() - aod_yrs.min())

    n_cname = nlight.columns
    nlight = nlight.set_index(n_cname[0])
    #------------------------------
    plt.subplot(321)
    sns.barplot(x = hcho.index, y = h_cname[1], data = hcho);
    #------------------------------
    plt.subplot(322)
    sns.pointplot(x = nlight.index, y = n_cname[1], data = nlight);
    # nlight[n_cname[1]].plot(kind = "area", colormap = "Spectral_r")
    #------------------------------
    plt.subplot(323)
    sns.barplot(x = aod.index, y = a_cname[1], data = aod);
    #------------------------------
    plt.subplot(324)
    h_spr, h_smr, h_fal, h_wtr = cal_seasons(hcho[h_cname[1]].values)
    h_ssn = combine_ssn(h_spr, h_smr, h_fal, h_wtr, 2005)
    sns.barplot(x = "years", y = "data", hue = "season", data = h_ssn);
    #------------------------------
    plt.subplot(325)
    df_yrs_h = pd.DataFrame([years, hcho_yrs], index = ["years", "data"]).T
    df_yrs_a = pd.DataFrame([years, aod_yrs], index = ["years", "data"]).T
    sns.pointplot(x = "years", y = "data", data = df_yrs_h)
    sns.pointplot(x = "years", y = "data", data = df_yrs_a)
    # sns.swarmplot(x = "years", y = "data", data = df_yrs_a);
    # sns.jointplot(df_yrs_h["data"], df_yrs_a["data"], kind="kde", size=7, space=0)
    print help(sns.jointplot)
    #------------------------------
    plt.subplot(326)
    a_spr, a_smr, a_fal, a_wtr = cal_seasons(aod[a_cname[1]].values)
    a_ssn = combine_ssn(a_spr, a_smr, a_fal, a_wtr, 2005)
    sns.barplot(x = "years", y = "data", hue = "season", data = a_ssn);
    # sns.stripplot(x = "season", y = "data", data = a_ssn, jitter = True);
    # sns.stripplot(x = "season", y = "data", data = h_ssn, jitter = True);
    #------------------------------


    plt.show()

def draw_before():
    import seaborn as sns
    import matplotlib.pyplot as plt
    wrk_dir = r"D:\hcho_change\Whole_China"
    hcho = pd.read_csv(os.path.join(wrk_dir, r"out\hcho_record.csv"))
    nlight = pd.read_csv(os.path.join(wrk_dir, r"out\nlight_record.csv"))
    aod = pd.read_csv(os.path.join(wrk_dir, r"out\aod_record.csv"))
    yrs = hcho.iloc[:, 0].values
    vals_hcho = hcho.iloc[:, 1].values
    x = np.arange(yrs.shape[0])
    ## subplot(nrow, ncols, plot_number)
    plt.subplot(3, 2, 1)
    hcho_yrs = cal_years(vals_hcho); hcho_yrs = pd.Series(hcho_yrs, index = np.linspace(0, 130, num = 11))
    plt.plot(x, vals_hcho, "-*")
    # help(hcho_yrs.plot)
    hcho_yrs.plot(kind = "line", colormap = "Pastel1")
    spr, smr, fal, wtr = cal_seasons(vals_hcho)
    plt.subplot(3, 2, 4)
    plt.plot(spr, "-*")
    plt.plot(smr, "-*")
    plt.plot(fal, "-*")
    plt.plot(wtr, "-*")
    vals = nlight.iloc[:, 1].values
    plt.subplot(3, 2, 2)
    plt.plot(vals, "-*")
    vals_aod = aod.iloc[:, 1].values
    plt.subplot(3, 2, 5)
    plt.plot(x, vals_aod, "-*")
    spr, smr, fal, wtr = cal_seasons(vals_aod)
    plt.subplot(3, 2, 6)
    # plt.plot(spr, "-*")
    # plt.plot(smr, "-*")
    # plt.plot(fal, "-*")
    # plt.plot(wtr, "-*")
    spr = pd.Series(spr)
    smr = pd.Series(smr)
    fal = pd.Series(fal)
    wtr = pd.Series(wtr)
    smr.plot(kind = "line", colormap = "Pastel1")
    spr.plot(kind = "area", colormap = "Spectral_r")
    fal.plot(kind = "bar", colormap = "CMRmap")
    wtr.plot(kind = "bar", colormap = "PiYG")
    help(wtr.plot)
    vals_aod_ = aod.iloc[:, 1]
    # help(vals_aod_.plot)
    plt.subplot(3, 2, 3)
    vals_aod_.plot(kind = "bar")

    # sns.jointplot(vals_hcho, vals_aod, kind="kde", size=7, space=0)
    sns.plt.show()
    plt.show()
    # print np.linspace(0, 130, num = 11)
"""

if __name__ == '__main__':
    main()
    print "OK, all done!"