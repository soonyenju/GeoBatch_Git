# coding: utf-8
import os
import arcpy

'''
author: SoonyenJu
Data: 2017-02-20
Version: 2nd Version
Contact: Soonyenju@foxmail.com
'''
def main():
    """
    this code is ready to use, only FOUR parameters need to be predined by users:
    1.path: path
    2.fixed file name: filename
    3.map title: title
    4.resolution: like (1753, 1241, 150)
    """
    path = r"D:\hcho_change\seasonal_series\out"
    symbologyLayer = r"D:\hcho_change\seasonal_series\season_no2\05n_1s.tif.lyr"
    filename = "w4"
    startyear = 2005 - 1
    title = "NO2 concentration distribution in winter ROI, "

    mxd = arcpy.mapping.MapDocument("CURRENT")
    mxd.saveACopy(path + "/" + "tempsave.mxd")
    # lyrs = arcpy.mapping.ListLayers(mxd)
    dfs = arcpy.mapping.ListDataFrames(mxd)[0]
    lyrs = arcpy.mapping.ListLayers(mxd, "", dfs)
    txtnum = len(arcpy.mapping.ListLayoutElements(mxd,"TEXT_ELEMENT"))

    for idx, lyr in enumerate(lyrs):
        print arcpy.mapping.ListLayers(mxd, lyr)[0]
        if lyrs[idx].name == "COI": continue
        elif lyrs[idx].name == "china": break
        lyr.visible = False
        arcpy.ApplySymbologyFromLayer_management (lyr, symbologyLayer)
        # arcpy.mapping.AddLayer(dfs,lyr,"TOP")
        lyr.visible = True
        arcpy.RefreshTOC()
        arcpy.RefreshActiveView()
        lyrname = arcpy.mapping.ListLayers(mxd, lyr)[0]
        fulldir = path + "/" + str(startyear + idx) + filename + ".jpg"
        """
        for i in range(txtnum):
            text_contrl = arcpy.mapping.ListLayoutElements(mxd,"TEXT_ELEMENT")[i]
            print text_contrl.text
            if text_contrl.text == "what text you want to change here":
                text_contrl.text = "text to replace"; mxd.save()
        """
        mxd.title = title + str(startyear + idx); mxd.save()

        arcpy.RefreshActiveView()
        # arcpy.mapping.ExportToJPEG(mxd, fulldir, "PAGE_LAYOUT", 1753, 1241, 150)
        arcpy.mapping.ExportToJPEG(mxd, fulldir, "PAGE_LAYOUT", 733, 467, 50)
        print "the " + str(idx + 1) + " pic has been output correctly."
        arcpy.mapping.RemoveLayer(dfs, lyrs[idx])

if __name__ == '__main__':
    main()
    print "ok, all done."