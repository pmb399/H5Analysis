# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d

# Data loaders
from .data_1d import load_1d
from .data_2d import load_2d
from .histogram import load_histogram

# Utilities
from .simplemath import apply_offset, apply_savgol, grid_data2d, grid_data, bin_data, grid_data_mesh
from .readutil import detector_norm

# Warnings
import warnings

def ScanAddition(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_item=None):
    """Internal function to handle scan addition.

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
    for i, (k, v) in enumerate(ScanData.items()):
        start_list.append(v.x_stream.min())
        end_list.append(v.x_stream.max())

        if i == 0: # Get the difference
            x_diff = np.abs(np.diff(v.x_stream).min())

    s = max(start_list)
    e = min(end_list)

    if s>=e:
        raise Exception("There is not sufficient overlap in x to perform interpolation.")
    
    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    max_steps = 104857600/8
    steps = int((e-s)/x_diff)

    if steps>max_steps:
        num = max_steps
    else:
        num = steps

    MASTER_x_stream = np.linspace(s,e,num)

    # Iterate over all loaded scans for interpolation
    for i, (k, v) in enumerate(ScanData.items()):
        # Set the first scan as master data
        if i == 0:
            MASTER_y_stream = interp1d(v.x_stream,v.y_stream)(MASTER_x_stream)
            name = str(k)+'+'
        else:
            # For additional scans, set the first x-scale as master and interpolate all
            # data suczessively to ensure appropriate addition
            interp = interp1d(v.x_stream, v.y_stream)(MASTER_x_stream)
            MASTER_y_stream += interp

            name += "_" + str(k)

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

def ScanSubtraction(config,file, x_stream, y_stream, minuend, subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_item=None):
    """ Internal function to handle scan subtraction.

        Parameters
        ----------
        args: Same as for the Load1d class, but
        minuend: list
            adds all scans in list, generates minuend
        subtrahend: list
            adds all scans in list, generates subtrahend
        kwargs: See Load1d class

        Returns
        -------
        data: dict
    """
    
    # Define generic object in which all data will be stored
    class added_object:
        def __init__(self):
            pass
        
    # Get the minuend and subtrahend data
    # Pass the scans specified in each list to the Scan addition function
    minuendData = ScanAddition(config,file, x_stream, y_stream, *minuend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None)
    subtrahendData = ScanAddition(config,file, x_stream, y_stream, *subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None)

    name = f"{minuend}-{subtrahend}"

    # Iterate over all loaded scans to determine bounds
    s = max(minuendData[0].x_stream.min(),subtrahendData[0].x_stream.min())
    e = min(minuendData[0].x_stream.max(),subtrahendData[0].x_stream.max())
    x_diff = np.abs(np.diff(minuendData[0].x_stream).min())

    if s>=e:
        raise Exception("There is not sufficient overlap in x to perform interpolation.")
    
    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    max_steps = 104857600/8
    steps = int((e-s)/x_diff)

    if steps>max_steps:
        num = max_steps
    else:
        num = steps

    MASTER_x_stream = np.linspace(s,e,num)

    # Define the first scan (addition of scans as master)
    MASTER_x_stream = np.linspace(s,e,num)
    MASTER_y_stream = interp1d(minuendData[0].x_stream, minuendData[0].y_stream)(MASTER_x_stream)

    # For additional scans, set the first x-scale as master and interpolate all
    # data suczessively to ensure appropriate addition
    interp = interp1d(subtrahendData[0].x_stream, subtrahendData[0].y_stream)(MASTER_x_stream)
    MASTER_y_stream -= interp

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

def ImageAddition(config, file, x_stream, detector, *args, norm=True, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None):
    """Internal function to handle image addition.

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
        
    # Load all 2d data to be added
    # Note that this is possible since load2d supports loading multiple scans
    ScanData = load_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=grid_x,grid_y=grid_y,norm_by=norm_by,)

    # Iterate over all loaded scans to determine bounds
    x_start_list = list()
    x_end_list = list()
    y_start_list = list()
    y_end_list = list()
    for i, (k, v) in enumerate(ScanData.items()):
        x_start_list.append(v.new_x.min())
        x_end_list.append(v.new_x.max())
        y_start_list.append(v.new_y.min())
        y_end_list.append(v.new_y.max())

        if i == 0: # Get the difference
            x_diff = np.abs(np.diff(v.new_x).min())
            y_diff = np.abs(np.diff(v.new_y).min())

    x_s = max(x_start_list)
    x_e = min(x_end_list)
    y_s = max(y_start_list)
    y_e = min(y_end_list)

    if x_s>=x_e:
        raise Exception("There is not sufficient overlap in x to perform interpolation.")
    
    if y_s>=y_e:
        raise Exception("There is not sufficient overlap in y to perform interpolation.")
    
    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    max_steps = 104857600/8
    x_steps = int((x_e-x_s)/x_diff)
    y_steps = int((y_e-y_s)/y_diff)

    if x_steps>max_steps:
        x_num = max_steps
    else:
        x_num = x_steps

    MASTER_x_stream = np.linspace(x_s,x_e,x_num)

    if y_steps>max_steps:
        y_num = max_steps
    else:
        y_num = y_steps

    MASTER_y_stream = np.linspace(y_s,y_e,y_num)

    # Iterate over all loaded scans
    for i, (k, v) in enumerate(ScanData.items()):
    # Set the first scan as master data
        if i == 0:
            MASTER_detector = interp2d(v.new_x,v.new_y,v.new_z)(MASTER_x_stream,MASTER_y_stream)
            name = str(k)+'+'
        else:            
            interp = interp2d(v.new_x,v.new_y,v.new_z)
            new_z = interp(MASTER_x_stream,MASTER_y_stream)

            MASTER_detector = np.add(MASTER_detector,new_z)
            
            name += "_" + str(k)

    # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
    data = dict()
    data[0] = added_object()
    data[0].new_x = MASTER_x_stream
    data[0].new_y = MASTER_y_stream
    data[0].new_z = MASTER_detector
    data[0].xmin = MASTER_x_stream.min()
    data[0].xmax = MASTER_x_stream.max()
    data[0].ymin = MASTER_y_stream.min()
    data[0].ymax = MASTER_y_stream.max()
    data[0].xlabel = x_stream
    data[0].ylabel = 'Scale'
    data[0].zlabel = detector
    data[0].filename = file

    data[0].scan = name

    # Apply x offset
    data[0].new_x = apply_offset(data[0].new_x, xoffset, xcoffset)

    # Apply y offset
    data[0].new_y = apply_offset(data[0].new_y, yoffset, ycoffset)

    # Normalize data to [0,1]
    if norm == True:
        data[0].new_z =  data[0].new_z / data[0].new_z.max()

    return data

def ImageSubtraction(config, file, x_stream, detector, minuend, subtrahend, norm=True, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None):

    """ Internal function to handle image subtraction.

        Parameters
        ----------
        args: Same as for the Load1d class, but
        minuend: list
            adds all images in list, generates minuend
        subtrahend: list
            adds all images in list, generates subtrahend
        kwargs: See Load1d class

        Returns
        -------
        data: dict
    """

   # Define generic object in which all data will be stored
    class added_object:
        def __init__(self):
            pass
    
    # Define minuend and subtrahend
    # Add images of all scans specified in respective lists,
    # then subtract
    minuend = ImageAddition(config, file, x_stream, detector, *minuend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=grid_x,grid_y=grid_y,norm_by=norm_by,)
    subtrahend = ImageAddition(config, file, x_stream, detector, *subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=grid_x,grid_y=grid_y,norm_by=norm_by,)

    name = f"{minuend}-{subtrahend}"

    # Set the master streams
    x_s = max(minuend[0].new_x.min(),subtrahend[0].new_x.min())
    x_e = min(minuend[0].new_x.max(),subtrahend[0].new_x.max())
    x_diff = np.abs(np.diff(minuend[0].new_x).min())
    y_s = max(minuend[0].new_y.min(),subtrahend[0].new_y.min())
    y_e = min(minuend[0].new_y.max(),subtrahend[0].new_y.max())
    y_diff = np.abs(np.diff(minuend[0].new_y).min())

    if x_s>=x_e:
        raise Exception("There is not sufficient overlap in x to perform interpolation.")

    if y_s>=y_e:
        raise Exception("There is not sufficient overlap in y to perform interpolation.")
    
    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    max_steps = 104857600/8
    x_steps = int((x_e-x_s)/x_diff)
    y_steps = int((y_e-y_s)/y_diff)

    if x_steps>max_steps:
        x_num = max_steps
    else:
        x_num = x_steps

    MASTER_x_stream = np.linspace(x_s,x_e,x_num)

    if y_steps>max_steps:
        y_num = max_steps
    else:
        y_num = y_steps

    MASTER_y_stream = np.linspace(y_s,y_e,y_num)
    MASTER_detector = interp2d(minuend[0].new_x,minuend[0].new_y,minuend[0].new_z)(MASTER_x_stream,MASTER_y_stream)

    # Interpolate the subtrahend onto minuend, then subtract
    interp = interp2d(subtrahend[0].new_x,subtrahend[0].new_y,subtrahend[0].new_z)
    new_z = interp(MASTER_x_stream,MASTER_y_stream)

    MASTER_detector = np.subtract(MASTER_detector,new_z)
    
    # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
    data = dict()
    data[0] = added_object()
    data[0].new_x = MASTER_x_stream
    data[0].new_y = MASTER_y_stream
    data[0].new_z = MASTER_detector
    data[0].xmin = MASTER_x_stream.min()
    data[0].xmax = MASTER_x_stream.max()
    data[0].ymin = MASTER_y_stream.min()
    data[0].ymax = MASTER_y_stream.max()
    data[0].xlabel = x_stream
    data[0].ylabel = 'Scale'
    data[0].zlabel = detector
    data[0].filename = file

    data[0].scan = name

    # Apply x offset
    data[0].new_x = apply_offset(data[0].new_x, xoffset, xcoffset)

    # Apply y offset
    data[0].new_y = apply_offset(data[0].new_y, yoffset, ycoffset)

    # Normalize data to [0,1]
    if norm == True:
        data[0].new_z =  data[0].new_z / data[0].new_z.max()

    return data

def HistogramAddition(config,file, x_stream, y_stream, z_stream, *args, norm=False):

    """Internal function to handle histogram addition.

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
    
    # Load all histograms, corresponding to the specified scans
    ScanData = load_histogram(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None)

    # Iterate over all loaded scans
    # Append all data arrays to the following lists
    x_data = list()
    y_data = list()
    z_data = list()
    for i, (k, v) in enumerate(ScanData.items()):
        x_data.append(v.x_data)
        y_data.append(v.y_data)
        z_data.append(v.z_data)

    # Combine the individual data arrays
    all_x = np.concatenate(tuple(x_data))
    all_y = np.concatenate(tuple(y_data))
    all_z = np.concatenate(tuple(z_data))
    
    # Mesh the data and generate 2d histogram
    xmin, xmax, ymin, ymax, xedge, yedge, new_z, zmin, zmax = grid_data_mesh(all_x,all_y,all_z)

    # Store data
    data = dict()
    data[0] = added_object()
    data[0].xmin = xmin
    data[0].xmax = xmax
    data[0].ymin = ymin
    data[0].ymax = ymax
    data[0].new_x = xedge
    data[0].new_y = yedge
    data[0].new_z = new_z
    data[0].zmin = zmin
    data[0].zmax = zmax
    data[0].x_data = all_x
    data[0].y_data = all_y
    data[0].z_data = all_z
    data[0].xlabel = x_stream
    data[0].ylabel = x_stream
    data[0].zlabel = z_stream
    data[0].filename = file

    if norm == True:
        data[0].new_z =  data[0].new_z / data[0].new_z.max()
        data[0].zmin = zmin / zmax
        data[0].zmax = 1

    return data