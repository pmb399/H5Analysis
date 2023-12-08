from .util import doesMatchPattern, check_key_in_dict, get_roi
from .ReadData import REIXS
from .simplemath import apply_offset, grid_data_mesh
import warnings
import numpy as np
from .parser import math_stream
from .spec_config import get_REIXSconfig


def loadMeshScans(file, x_stream, y_stream, z_stream, *args, norm=True, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None):
    """Internal function to generate scatter plots for (x,y,z) SCA data"""

    def get_data(req, data, arg):

        try:
            strip_roi = req.split("[")[1].rstrip("]")
            roi_low, roi_high = get_roi(strip_roi)
            roi = (roi_low,roi_high)
            req = req.split("[")[0]
        except:
            roi = None

        if check_key_in_dict(req,REIXSConfig):
            data[arg].Scan(req,roi)
            return data[arg].data
        else:
            try:
                # Else, load from pandas SCA data frame
                return np.array(data[arg].sca_data[req])
            except:
                raise UserWarning("Special Stream not defined.")

    # Generate dictionary to store data with REIXS objects
    data = dict()
    for arg in args:
        # Load scans to dict
        data[arg] = REIXSobj = REIXS(file,arg)
        data[arg].scan = arg

        REIXSConfig = get_REIXSconfig()

        # Assign the calculated result to the y_stream of the object in data dict
        data[arg].y_data = math_stream(y_stream, data, arg, get_data)
        data[arg].y_data = apply_offset(data[arg].y_data, yoffset, ycoffset)

        # Assign the calculated result to the x_stream of the object in data dict
        data[arg].x_data = math_stream(x_stream, data, arg, get_data)
        data[arg].x_data = apply_offset(data[arg].x_data, xoffset, xcoffset)

        # Aplly simple math to x-stream
        data[arg].z_data = math_stream(z_stream, data, arg, get_data)

        # Normalize if requested
        if norm == True:
            data[arg].z_data = np.interp(
                data[arg].z_data, (data[arg].z_data.min(), data[arg].z_data.max()), (0, 1))

        xmin, xmax, ymin, ymax, xedge, yedge, new_z, zmin, zmax = grid_data_mesh(data[arg].x_data,data[arg].y_data,data[arg].z_data)
        data[arg].xmin = xmin
        data[arg].xmax = xmax
        data[arg].ymin = ymin
        data[arg].ymax = ymax
        data[arg].xedge = xedge
        data[arg].yedge = yedge
        data[arg].new_z = new_z
        data[arg].zmin = zmin
        data[arg].zmax = zmax

    return data
