from .util import doesMatchPattern, check_key_in_dict, get_roi
from .ReadData import REIXS
from .spec_config import get_REIXSconfig
from .simplemath import apply_offset, grid_data, apply_savgol, bin_data
import warnings
import numpy as np
from .parser import math_stream
from shapely.geometry import Point, Polygon

def loadSCAscans(file, x_stream, y_stream, *args, norm=True, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, energyloss=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_items={}, background = None):
    """Internal function to load and process data to 1d
    
        Parameters
        ----------
        See Load1d function.
    """

    def get_y_data(req, data, arg):

        try:
            strip_roi = req.split("[")[1].rstrip("]")
            roi_low, roi_high = get_roi(strip_roi)
            roi = (roi_low,roi_high)
            req = req.split("[")[0]
        except:
            roi = None

        if check_key_in_dict(req,REIXSConfig):
            data[arg].Scan(req,roi,kwargs)
            return data[arg].data
        else:
            try:
                # Else, load from pandas SCA data frame
                return np.array(data[arg].sca_data[req])
            except:
                raise UserWarning("Special Stream not defined.")
            
    def get_x_data(req, data, arg):
        try:
            if check_key_in_dict(req,REIXSConfig):
                data[arg].Scan(req,None)
                return np.array(data[arg].data)
            else:
                return np.array(data[arg].sca_data[req])
        except Exception as e:
           return np.array(range(0, len(data[arg].y_stream)), dtype=int)


    # Add all data (REIXS objects) to dictionary
    data = dict()
    
    # Iterate over all scans requested in load call
    for arg in args:
        data[arg] = REIXSobj = REIXS(file,arg)
        data[arg].scan = arg

        REIXSConfig = get_REIXSconfig()

        kwargs = dict()
        kwargs['background'] = background
        data[arg].y_stream = math_stream(y_stream, data, arg, get_y_data)
        data[arg].x_stream = math_stream(x_stream, data, arg, get_x_data)
        
        # Get legend items
        try:
            data[arg].legend = legend_items[arg]
        except:
            data[arg].legend = f"S{arg}_{y_stream}"

        #Bin the data if requested
        if binsize != None:
            data[arg].x_stream, data[arg].y_stream = bin_data(data[arg].x_stream,data[arg].y_stream,binsize)

        # Grid the data if specified
        if grid_x != [None, None, None]:
            new_x, new_y = grid_data(
                data[arg].x_stream, data[arg].y_stream, grid_x)

            data[arg].x_stream = new_x
            data[arg].y_stream = new_y

        # Apply offsets to x-stream
        data[arg].x_stream = apply_offset(
        data[arg].x_stream, xoffset, xcoffset)

        # Apply normalization to [0,1]
        if norm == True:
            data[arg].y_stream = np.interp(
                data[arg].y_stream, (data[arg].y_stream.min(), data[arg].y_stream.max()), (0, 1))

        # Apply offset to y-stream
        data[arg].y_stream = apply_offset(
        data[arg].y_stream, yoffset, ycoffset)
               
        # Smooth and take derivatives
        if savgol != None:
            if isinstance(savgol,tuple):
                if len(savgol) == 2: # Need to provide window length and polynomial order
                    savgol_deriv = 0 # Then, no derivative is taken
                elif len(savgol) == 3:
                    savgol_deriv = savgol[2] # May also specify additional argument for derivative order
                else:
                    raise TypeError("Savgol smoothing arguments incorrect.")
                data[arg].x_stream, data[arg].y_stream = apply_savgol(data[arg].x_stream,data[arg].y_stream,savgol[0],savgol[1],savgol_deriv)

                if norm == True:
                    data[arg].y_stream = data[arg].y_stream / \
                    data[arg].y_stream.max()
            else:
                raise TypeError("Savgol smoothing arguments incorrect.")

        # Transforms RIXS to energy loss scale if incident energy is given
        if energyloss != None:
            # If True, use value from mono to transform to energy loss, else use manual float input
            if energyloss == True:
                data[arg].Scan('Mono Energy',None)
                data[arg].x_stream = np.average(data[arg].data)-data[arg].x_stream
            else:
                data[arg].x_stream = energyloss-data[arg].x_stream

    return data