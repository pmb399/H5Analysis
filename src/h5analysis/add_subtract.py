"""Adds support for addition and subtraction of scans"""

# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d

# Data loaders
from .data_1d import load_1d, apply_kwargs_1d
from .data_2d import load_2d, apply_kwargs_2d
from .data_3d import load_3d
from .histogram import load_histogram

# Warnings
import warnings

def ScanAddition(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_item=None, twin_y=False, matplotlib_props=dict()):
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
    ScanData = load_1d(config,file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=binsize)

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
    steps = int((e-s)/x_diff)+1

    if steps>max_steps:
        num = max_steps
    else:
        num = steps

    MASTER_x_stream = np.linspace(s,e,num)

    # Keep track of axis labels
    xaxis_label = list()
    yaxis_label = list()
    # Iterate over all loaded scans for interpolation
    for i, (k, v) in enumerate(ScanData.items()):
        for x in v.xaxis_label:
            xaxis_label.append(x)
        for y in v.yaxis_label:
            yaxis_label.append(y)
        # Set the first scan as master data
        if i == 0:
            MASTER_y_stream = interp1d(v.x_stream,v.y_stream)(MASTER_x_stream)
            name = str(k)
        else:
            # For additional scans, set the first x-scale as master and interpolate all
            # data suczessively to ensure appropriate addition
            interp = interp1d(v.x_stream, v.y_stream)(MASTER_x_stream)
            MASTER_y_stream += interp

            name += "+" + str(k)

    # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
    data = dict()
    data[0] = added_object()
    data[0].xlabel = x_stream
    data[0].ylabel = y_stream
    data[0].xaxis_label = xaxis_label
    data[0].yaxis_label = yaxis_label
    data[0].filename = file
    data[0].x_stream = MASTER_x_stream
    data[0].y_stream = MASTER_y_stream
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

def ScanSubtraction(config,file, x_stream, y_stream, minuend, subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_item=None, twin_y=False, matplotlib_props=dict()):
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

    # Make sure minuend and subtrahend are of type list (even if only one scan is passed)
    if isinstance(minuend,int):
        minuend = [minuend]
    if isinstance(subtrahend,int):
        subtrahend = [subtrahend]
        
    # Get the minuend and subtrahend data
    # Pass the scans specified in each list to the Scan addition function
    minuendData = ScanAddition(config,file, x_stream, y_stream, *minuend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=binsize)
    subtrahendData = ScanAddition(config,file, x_stream, y_stream, *subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=binsize)

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
    steps = int((e-s)/x_diff)+1

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
    data[0].xaxis_label = minuendData[0].xaxis_label + subtrahendData[0].xaxis_label
    data[0].yaxis_label = minuendData[0].yaxis_label + subtrahendData[0].yaxis_label
    data[0].filename = file
    data[0].x_stream = MASTER_x_stream
    data[0].y_stream = MASTER_y_stream
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

def ImageAddition_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None,binsize_x=None,binsize_y=None):
    """Internal function to handle image addition.

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
    ScanData = load_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=grid_x,grid_y=grid_y,norm_by=norm_by,binsize_x=binsize_x,binsize_y=binsize_y)

    return ImageAddition(ScanData, file, x_stream, detector, *args, norm=norm, xoffset=xoffset, xcoffset=xcoffset, yoffset=yoffset, ycoffset=ycoffset)

def ImageSubtraction_2d(config, file, x_stream, detector, minuend, subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None,binsize_x=None,binsize_y=None):
    """Internal function to handle image subtraction.

            Parameters
            ----------
            args: Same as for the Load2d class
            kwargs: See Load2d class

            Returns
            -------
            data: dict
        """   
    
    # Make sure minuend and subtrahend are of type list (even if only one scan is passed)
    if isinstance(minuend,int):
        minuend = [minuend]
    if isinstance(subtrahend,int):
        subtrahend = [subtrahend]

    # Define minuend and subtrahend
    # Add images of all scans specified in respective lists,
    # then subtract
    minuendData = ImageAddition_2d(config, file, x_stream, detector, *minuend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=grid_x,grid_y=grid_y,norm_by=norm_by,binsize_x=binsize_x,binsize_y=binsize_y)
    subtrahendData = ImageAddition_2d(config, file, x_stream, detector, *subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=grid_x,grid_y=grid_y,norm_by=norm_by,binsize_x=binsize_x,binsize_y=binsize_y)

    return ImageSubtraction(minuendData, subtrahendData, file, x_stream, detector, minuend, subtrahend, norm=norm, xoffset=xoffset, xcoffset=xcoffset, yoffset=yoffset, ycoffset=ycoffset)

def ImageAddition_hist(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=None, binsize_y=None):
    """Internal function to handle image addition.

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

    return ImageAddition(ScanData, file, x_stream, z_stream, *args, norm=norm, xoffset=xoffset, xcoffset=xcoffset, yoffset=yoffset, ycoffset=ycoffset)

def ImageSubtraction_hist(config, file, x_stream, y_stream, z_stream, minuend, subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=None, binsize_y=None):
    """Internal function to handle image subtraction.

            Parameters
            ----------
            args: Same as for the Load2d class
            kwargs: See Load2d class

            Returns
            -------
            data: dict
        """   
    
    # Make sure minuend and subtrahend are of type list (even if only one scan is passed)
    if isinstance(minuend,int):
        minuend = [minuend]
    if isinstance(subtrahend,int):
        subtrahend = [subtrahend]

    # Define minuend and subtrahend
    # Add images of all scans specified in respective lists,
    # then subtract
    minuendData = ImageAddition_hist(config, file, x_stream, y_stream, z_stream, *minuend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=binsize_x, binsize_y=binsize_y)
    subtrahendData = ImageAddition_hist(config, file, x_stream, y_stream, z_stream, *subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=binsize_x, binsize_y=binsize_y)

    return ImageSubtraction(minuendData, subtrahendData, file, x_stream, z_stream, minuend, subtrahend, norm=norm, xoffset=xoffset, xcoffset=xcoffset, yoffset=yoffset, ycoffset=ycoffset)


def ImageAddition(ScanData, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None):
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
        
    # Iterate over all loaded scans to determine bounds
    x_start_list = list()
    x_end_list = list()
    y_start_list = list()
    y_end_list = list()
    x_list = list()
    y_list = list()
    for i, (k, v) in enumerate(ScanData.items()):
        x_start_list.append(v.new_x.min())
        x_end_list.append(v.new_x.max())
        y_start_list.append(v.new_y.min())
        y_end_list.append(v.new_y.max())
        x_list.append(v.new_x)
        y_list.append(v.new_y)

        if i == 0: # Get the difference
            x_comp = v.new_x
            y_comp = v.new_y
            x_diff = np.abs(np.diff(v.new_x).min())
            y_diff = np.abs(np.diff(v.new_y).min())

    #Check if we need to interpolate scales
    if all([np.array_equal(x_comp,test_scale) for test_scale in x_list]) and all([np.array_equal(y_comp,test_scale) for test_scale in y_list]):
        # All scales are equal, no interpolation required
        MASTER_x_stream = x_comp
        MASTER_y_stream = y_comp

        # Iterate over all loaded scans
        for i, (k, v) in enumerate(ScanData.items()):
        # Set the first scan as master data
            if i == 0:
                MASTER_detector = v.new_z
                name = str(k)
            else:            
                MASTER_detector = np.add(MASTER_detector,v.new_z)
                name += "+" + str(k)

    else:
        # Interpolation required
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
        x_steps = int((x_e-x_s)/x_diff)+1
        y_steps = int((y_e-y_s)/y_diff)+1

        if x_steps*y_steps>max_steps:
            step_norm = int(np.ceil(np.sqrt(x_steps*y_steps/13107200)))
            x_num = int(x_steps/step_norm)
            y_num = int(y_steps/step_norm)
        else:
            x_num = x_steps
            y_num = y_steps

        MASTER_x_stream = np.linspace(x_s,x_e,x_num)
        MASTER_y_stream = np.linspace(y_s,y_e,y_num)

        # Iterate over all loaded scans
        for i, (k, v) in enumerate(ScanData.items()):
        # Set the first scan as master data
            if i == 0:
                MASTER_detector = interp2d(v.new_x,v.new_y,v.new_z)(MASTER_x_stream,MASTER_y_stream)
                name = str(k)
            else:            
                interp = interp2d(v.new_x,v.new_y,v.new_z)
                new_z = interp(MASTER_x_stream,MASTER_y_stream)

                MASTER_detector = np.add(MASTER_detector,new_z)
                
                name += "+" + str(k)

    # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
    data = dict()
    data[0] = added_object()
    data[0].new_x = MASTER_x_stream
    data[0].new_y = MASTER_y_stream
    data[0].new_z = MASTER_detector
    data[0].xlabel = x_stream
    data[0].ylabel = 'Scale'
    data[0].zlabel = detector
    data[0].filename = file

    data[0].scan = name

    # Apply kwargs
    data[0].new_x,data[0].new_y,data[0].new_z = apply_kwargs_2d(data[0].new_x,data[0].new_y,data[0].new_z,norm,xoffset,xcoffset,yoffset,ycoffset,None,None)

    return data

def ImageSubtraction(minuend, subtrahend, file, x_stream, detector, str_minuend, str_subtrahend, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None):

    """ Internal function to handle image subtraction.

        Parameters
        ----------
        args: Same as for the Load1d class, but
        minuend: list
            adds all images in list, generates minuend
        subtrahend: list
            adds all images in list, generates subtrahend
        kwargs: See Load2d class

        Returns
        -------
        data: dict
    """

   # Define generic object in which all data will be stored
    class added_object:
        def __init__(self):
            pass
    
    name = f"{str_minuend}-{str_subtrahend}"

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
    x_steps = int((x_e-x_s)/x_diff)+1
    y_steps = int((y_e-y_s)/y_diff)+1

    if x_steps*y_steps>max_steps:
        step_norm = int(np.ceil(np.sqrt(x_steps*y_steps/13107200)))
        x_num = int(x_steps/step_norm)
        y_num = int(y_steps/step_norm)
    else:
        x_num = x_steps
        y_num = y_steps

    MASTER_x_stream = np.linspace(x_s,x_e,x_num)
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
    data[0].xlabel = x_stream
    data[0].ylabel = 'Scale'
    data[0].zlabel = detector
    data[0].filename = file

    data[0].scan = name

    # Apply kwargs
    data[0].new_x,data[0].new_y,data[0].new_z = apply_kwargs_2d(data[0].new_x,data[0].new_y,data[0].new_z,norm,xoffset,xcoffset,yoffset,ycoffset,None,None)

    return data

def StackAddition(config, file, ind_stream, stack, *args,**kwargs):

    """
    Adds multiple image stacks, given the scales are identical.

    Parameters
    ----------
    arguments: Same as for the Load1d class, but
    *args: multiple scans, comma separated
    kwargs: See Load3d class

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

    
    # Retrieve all data
    stack_data = list()
    stack_data_dims = list()
    for arg in args:
        obj = load_3d(config,file,ind_stream,stack,arg,**kwargs)[arg]
        stack_data.append(obj)
        stack_data_dims.append(np.shape(obj.stack))

    # make sure the dimensions are matching
    if all([np.array_equal(stack_data_dims[0],dims) for dims in stack_data_dims]):
        pass
    else:
        raise Exception("Adding stacks with incompatible dimensions.")
    
    data = dict()
    data[0] = added_object()

    for i,s in enumerate(stack_data):
        if i == 0:
            MASTER_STACK = s.stack
            filename = s.filename
            xlabel = s.xlabel
            ylabel = s.ylabel
            zlabel = s.zlabel

        else:
            MASTER_STACK = np.add(MASTER_STACK,s.stack)

    # Generate 3d stack from gridded z-data in stack_grid list
    # Store all data in dict
    data[0].stack = MASTER_STACK
    data[0].ind_stream = stack_data[0].ind_stream
    data[0].str_ind_stream = stack_data[0].str_ind_stream
    data[0].new_x = stack_data[0].new_x
    data[0].new_y = stack_data[0].new_y

    data[0].scan = 0
    data[0].filename = filename
    data[0].xlabel = xlabel
    data[0].ylabel = ylabel
    data[0].zlabel = zlabel


    return data

def StackSubtraction(config, file, ind_stream, stack, minuend, subtrahend, **kwargs):
    """
    Subtracts multiple image stacks, given the scales are identical.

    Parameters
    ----------
    args: Same as for the Load1d class, but
    minuend: list
        adds all images in list, generates minuend
    subtrahend: list
        adds all images in list, generates subtrahend
    kwargs: See Load3d class

    Returns
    -------
    data: dict
    """

    # Define generic object in which all data will be stored
    class added_object:
        def __init__(self):
            pass

    minuend = StackAddition(config, file, ind_stream, stack, *minuend,**kwargs)
    subtrahend = StackAddition(config, file, ind_stream, stack, *subtrahend,**kwargs)
    
    # make sure the dimensions are matching
    if np.shape(minuend[0].stack) == np.shape(subtrahend[0].stack):
        pass
    else:
        raise Exception("Subtracting stacks with incompatible dimensions.")
    
    data = dict()
    data[0] = added_object()

    MASTER_STACK = np.subtract(minuend[0].stack,subtrahend[0].stack)

    # Generate 3d stack from gridded z-data in stack_grid list
    # Store all data in dict
    data[0].stack = MASTER_STACK
    data[0].ind_stream = minuend[0].ind_stream
    data[0].str_ind_stream = minuend[0].str_ind_stream
    data[0].new_x = minuend[0].new_x
    data[0].new_y = minuend[0].new_y

    data[0].scan = 0
    data[0].filename = minuend[0].filename
    data[0].xlabel = minuend[0].xlabel
    data[0].ylabel = minuend[0].ylabel
    data[0].zlabel = minuend[0].zlabel

    return data