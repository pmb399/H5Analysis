"""Stitches multiple scans of data together, averages the overlap."""

# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d

# Data loaders
from .data_1d import load_1d, apply_kwargs_1d
from .data_2d import load_2d, apply_kwargs_2d
from .histogram import load_histogram

def ScanStitch(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_item=None, twin_y=False, matplotlib_props=dict()):
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
    ScanData = load_1d(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=binsize)

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

    # Keep track of axis labels
    xaxis_label = list()
    yaxis_label = list()

    # Iterate over all loaded scans for interpolation
    for i, (k, v) in enumerate(ScanData.items()):
        for x in v.xaxis_label:
            xaxis_label.append(x)
        for y in v.yaxis_label:
            yaxis_label.append(y)
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
    data[0].xaxis_label = xaxis_label
    data[0].yaxis_label = yaxis_label
    data[0].scan = name
    data[0].twin_y = twin_y

    # Get legend items
    if legend_item != None:
        data[0].legend = legend_item
    else:
        data[0].legend = f"{config.index}-S{name}_{x_stream}_{y_stream}"

    # Set matplotlib props
    data[0].matplotlib_props = matplotlib_props

    # Apply kwargs
    data[0].x_stream,data[0].y_stream = apply_kwargs_1d(data[0].x_stream,data[0].y_stream,norm,xoffset,xcoffset,yoffset,ycoffset,grid_x,savgol,None)

    return data

def ImageStitch_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,norm_by=None,average=False, binsize_x=None,binsize_y=None):
    """Internal function to handle image stitching in 2d.

            Parameters
            ----------
            args: Same as for the Load2d class
            kwargs: See Load2d class

            Returns
            -------
            data: dict
        """

    # Load all 2d data to be added
    # Note that this is possible since load2d supports loading multiple scans
    ScanData = load_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,norm_by=norm_by,binsize_x=binsize_x,binsize_y=binsize_y)

    return ImageStitch(ScanData, file, x_stream, detector, *args, norm=norm, xoffset=xoffset, xcoffset=xcoffset, yoffset=yoffset, ycoffset=ycoffset,average=average)

def ImageStitch_hist(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, average=False, binsize_x=None, binsize_y=None):
    """Internal function to handle image stitching for histogram.

            Parameters
            ----------
            args: Same as for the Load2d class
            kwargs: See Load2d class

            Returns
            -------
            data: dict
        """

    # Load all 2d data to be added
    # Note that this is possible since load2d supports loading multiple scans
    ScanData = load_histogram(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=binsize_x, binsize_y=binsize_y)

    return ImageStitch(ScanData, file, x_stream, z_stream, *args, norm=norm, xoffset=xoffset, xcoffset=xcoffset, yoffset=yoffset, ycoffset=ycoffset, average=average)


def ImageStitch(ScanData, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,average=False):
    """Internal function to handle image stitching.

        Parameters
        ----------
        args: Same as for the Load2d class
        kwargs: See Load2d class

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

    # Start by getting dimensions
    min_x = list()
    max_x = list()
    min_y = list()
    max_y = list()
    diff_x = list()
    diff_y = list()

    for k,v in ScanData.items():
        min_x.append(min(v.new_x))
        max_x.append(max(v.new_x))
        min_y.append(min(v.new_y))
        max_y.append(max(v.new_y))
        diff_x.append(np.abs(np.diff(v.new_x)).min())
        diff_y.append(np.abs(np.diff(v.new_y)).min())

    # Determine corners
    lower_x = min(min_x)
    upper_x = max(max_x)
    lower_y = min(min_y)
    upper_y = max(max_y)
    min_diff_x = min(diff_x)
    min_diff_y = min(diff_y)

    # Determine number of points from differences
    numPoints_x = int(np.ceil((upper_x-lower_x)/min_diff_x))
    numPoints_y = int(np.ceil((upper_y-lower_y)/min_diff_y))


    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    max_steps = 104857600/8

    if numPoints_x*numPoints_y>max_steps:
        step_norm = int(np.ceil(np.sqrt(numPoints_x*numPoints_y/max_steps)))
        x_num = int(numPoints_x/step_norm)
        y_num = int(numPoints_y/step_norm)
    else:
        x_num = numPoints_x
        y_num = numPoints_y

    # Generate new scales
    new_x = np.linspace(lower_x,upper_x,x_num)
    new_y = np.linspace(lower_y,upper_y,y_num)

    if average == False:
        for i,(k,v) in enumerate(ScanData.items()):
            if i ==0:
                # Interpolate image on new big image
                matrix = interp2d(v.new_x,v.new_y,v.new_z,fill_value=np.nan)(new_x,new_y)
            else:
                # Do this for all images
                # Check if no information (NaN) has been added to the composite image, if so, add - else, set addition to 0
                matrix2 = interp2d(v.new_x,v.new_y,v.new_z,fill_value=np.nan)(new_x,new_y)
                matrix_nan = np.isnan(matrix)
                matrix2_nan = np.isnan(matrix2)
                m_keep_matrix2 = matrix_nan & ~matrix2_nan
                matrix[m_keep_matrix2] = matrix2[m_keep_matrix2]

    else:
        for i,(k,v) in enumerate(ScanData.items()):
            if i ==0:
                # Interpolate image on new big image
                # Initilize divisor to track how many contributions per data points
                # Set the contribution to 1 where added, else 0 - use array masking
                # Add the new contributions to divisor
                matrix = interp2d(v.new_x,v.new_y,v.new_z,fill_value=np.nan)(new_x,new_y)
                divisor = np.zeros_like(matrix)
                ones = np.ones_like(matrix)
                ones[np.isnan(matrix)] = 0
                divisor = np.add(divisor,ones)
            else:
                # Same as above
                matrix2 = interp2d(v.new_x,v.new_y,v.new_z,fill_value=np.nan)(new_x,new_y)
                matrix = np.nansum(np.dstack((matrix,matrix2)),2)
                ones = np.ones_like(matrix)
                ones[np.isnan(matrix2)] = 0
                divisor = np.add(divisor,ones)

        # Divide big matrix by divisor to get average
        matrix = np.true_divide(matrix,divisor,where=divisor!=0)

    # Remove NaN values and set to 0
    matrix = np.nan_to_num(matrix,nan=0,posinf=0,neginf=0)

    # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
    data = dict()
    data[0] = added_object()
    data[0].new_x = new_x
    data[0].new_y = new_y
    data[0].new_z = matrix
    data[0].xlabel = x_stream
    data[0].ylabel = 'Scale'
    data[0].zlabel = detector
    data[0].filename = file

    data[0].scan = args

    # Apply kwargs
    data[0].new_x,data[0].new_y,data[0].new_z = apply_kwargs_2d(data[0].new_x,data[0].new_y,data[0].new_z,norm,xoffset,xcoffset,yoffset,ycoffset,None,None)

    return data