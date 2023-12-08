import numpy as np
from .ReadData import REIXS
from .simplemath import apply_offset, grid_data2d
from .parser import math_stream
from .util import check_key_in_dict
from .spec_config import get_REIXSconfig
from .readutil import detector_norm

def loadMCAscans(file, x_stream, detector, *args, norm=True, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None], energyloss=False, norm_by=None, background=None):
    """Internal function to load MCA data
    
        Parameters
        ----------
        See Load2d function.
    """

    def get_z_data(detector, data, arg):
        data[arg].Scan(detector,None,kwargs)
        return data[arg].mca
    
    def get_y_data(detector, data, arg):
        data[arg].Scan(detector,None)
        return data[arg].data_scale
    
    def get_x_data(req, data, arg):
        # Sets the independent axis (x-axis values)
        try:
            if check_key_in_dict(req,REIXSConfig):
                data[arg].Scan(req,None)
                return np.array(data[arg].data)
            else:
                return np.array(data[arg].sca_data[req])
        except Exception as e:
           return np.array(range(0, len(data[arg].y_stream)), dtype=int)
    
    # Place all loaded REIXS objects in data dictionary
    data = dict()
    for arg in args:

        data[arg] = REIXSobj = REIXS(file,arg)
        data[arg].scan = arg

        REIXSConfig = get_REIXSconfig()
        kwargs = dict()
        kwargs['background'] = background


        # Assign the calculated result to the y_stream of the object in data dict
        # May apply math operations and offsets
        data[arg].y_data = get_y_data(detector, data, arg)
        data[arg].y_data = apply_offset(data[arg].y_data, yoffset, ycoffset)

        # Assign the calculated result to the x_stream of the object in data dict
        # May apply math operations and offsets
        data[arg].x_data = math_stream(x_stream, data, arg, get_x_data)
        data[arg].x_data = apply_offset(data[arg].x_data, xoffset, xcoffset)

        # Apply simple math operations on 2D MCA detector data
        data[arg].detector = math_stream(
            detector, data, arg, get_z_data)

        # Normalize MCA data by SCA
        if not isinstance(norm_by,type(None)):
            if check_key_in_dict(norm_by,REIXSConfig):
                data[arg].Scan(norm_by,None)
                normalization = data[arg].data
            else:
                try:
                # Else, load from pandas SCA data frame
                    normalization = np.array(data[arg].sca_data[norm_by])
                except:
                    raise UserWarning("Special Stream not defined.")

            data[arg].detector = detector_norm(data[arg].detector,normalization)

        # Normalize if requested
        if norm == True:
            data[arg].detector = data[arg].detector/np.max(data[arg].detector)

        xmin, xmax, ymin, ymax, new_x, new_y, new_z = grid_data2d(data[arg].x_data, data[arg].y_data, data[arg].detector, grid_x=grid_x,grid_y=grid_y,energyloss=energyloss)
        data[arg].xmin = xmin
        data[arg].xmax = xmax
        data[arg].ymin = ymin
        data[arg].ymax = ymax
        data[arg].new_x = new_x
        data[arg].new_y = new_y
        data[arg].new_z = new_z

    return data
