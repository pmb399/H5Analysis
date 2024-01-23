# Scientific modules
import numpy as np
from scipy.interpolate import interp1d

# Data loaders
from .data_1d import load_1d

# Utilities
from .simplemath import apply_offset, apply_savgol, grid_data2d, grid_data, bin_data, grid_data_mesh

def ScanStitch(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_item=None):
    """Internal function to handle scan stitching.

        Parameters
        ----------
        args: Same as for the Load1d class
        kwargs: See Load1d class

        Returns
        -------
        data: dict
    """

    # Define generic object in which all data will be stored
    class added_object:
        def __init__(self):
            pass

    # Ensure we only add a unique scan once
    for i in args:
        if args.count(i) > 1:
            raise ValueError("Cannot add the same scan to itself")

    # Load all specified scan data
    ScanData = load_1d(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None)

    # Iterate over all loaded scans to determine bounds
    start_list = list()
    end_list = list()
    diff_list = list()

    for i, (k, v) in enumerate(ScanData.items()):
        start_list.append(v.x_stream.min())
        end_list.append(v.x_stream.max())
        diff_list.append(np.abs(np.diff(v.x_stream).min()))

    s = min(start_list)
    e = max(end_list)
    x_diff = min(diff_list)
  
    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    max_steps = 104857600/8
    steps = int((e-s)/x_diff)+1

    if steps>max_steps:
        num = max_steps
    else:
        num = steps

    MASTER_x_stream = np.linspace(s,e,num)

    # Store all y values
    MASTER_y_list = list() # y-arrays interpolated to common scale
    MASTER_y_nan_list = list() # where nan values are stored

    # Iterate over all loaded scans for interpolation
    for i, (k, v) in enumerate(ScanData.items()):
        # interpolate to common scale
        item = interp1d(v.x_stream,v.y_stream,bounds_error=False)(MASTER_x_stream)
        # Store results
        MASTER_y_list.append(item)
        # Return boolean True where array element is a number
        MASTER_y_nan_list.append(~np.isnan(item))

        if i == 0:
            name = str(k)+'+'
        else:
            name += "_" + str(k)

    # This is for averaging
    # Sum the arrays of common length, treat nan as 0
    # For each element, sum how many True (numbers) contribute to the sum
    # Normalize to get average by array division
    MASTER_y_stream = np.nansum(MASTER_y_list,axis=0)/np.sum(MASTER_y_nan_list,axis=0)

    # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
    data = dict()
    data[0] = added_object()
    data[0].xlabel = x_stream
    data[0].ylabel = y_stream
    data[0].filename = file
    data[0].x_stream = MASTER_x_stream
    data[0].y_stream = MASTER_y_stream
    data[0].scan = name

    # Get legend items
    if legend_item != None:
        data[0].legend = legend_item
    else:
        data[0].legend = f"S{name}_{y_stream}"

    # Normalize data to [0,1]
    if norm == True:
        data[0].y_stream = np.interp(
            data[0].y_stream, (data[0].y_stream.min(), data[0].y_stream.max()), (0, 1))
        
    # Grid the data if specified
    if grid_x != [None, None, None]:
        new_x, new_y = grid_data(
            data[0].x_stream, data[0].y_stream, grid_x)

        data[0].x_stream = new_x
        data[0].y_stream = new_y

    # May apply constant and polynomial offset
    data[0].x_stream = apply_offset(data[0].x_stream, xoffset, xcoffset)
    data[0].y_stream = apply_offset(data[0].y_stream, yoffset, ycoffset)

    #Bin the data if requested
    if binsize != None:
        data[0].x_stream, data[0].y_stream = bin_data(data[0].x_stream,data[0].y_stream,binsize)

    # Apply smoothing and derivatives
    if savgol != None:
        if isinstance(savgol,tuple):
            if len(savgol) == 2:
                savgol_deriv = 0
            elif len(savgol) == 3:
                savgol_deriv = savgol[2]
            else:
                raise TypeError("Savgol smoothing arguments incorrect.")
            data[0].x_stream, data[0].y_stream = apply_savgol(data[0].x_stream,data[0].y_stream,savgol[0],savgol[1],savgol_deriv)

            if norm == True:
                data[0].y_stream = data[0].y_stream / \
                data[0].y_stream.max()
        else:
            raise TypeError("Savgol smoothing arguments incorrect.")

    return data

