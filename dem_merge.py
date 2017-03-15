# Import system modules
import arcpy
from arcpy import env
from arcpy.sa import *

# Set environment settings
env.workspace = r"D:\hcho_change\DEM"

# Set local variables
rasters = arcpy.ListRasters("*", "tif")

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")

# Execute Combine
out = Combine([rasters])

# Save the output
# out.save("C:/sapyexamples/output/outcombine")