# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.signal import savgol_filter

# Warnings
import warnings

def apply_offset(stream, offset=None, coffset=None):
    """Apply constant or polynomial offset to specified stream
    
        Parameters
        ----------
        stream : array
            Specify the data to act on
        offset : list of tuples
            List all tuples with shift values (is,should)
        cofset : float
            Shift by constant

        Returns
        -------
        stream: numpy array
            offset array
    """

    # Do the polynomial fitting with deg = len(list)-1
    if offset != None:
        offsetarray = np.array(offset)
        # Limit the applicable shift to be quadratic or of lower order
        deg = min(2,len(offsetarray)-1)
        coeff = np.polyfit(
            offsetarray[:, 0], offsetarray[:, 1], deg=deg)

        # Make sure that constant shift is applied if only one tuple provided (handled as offset)
        if len(coeff) == 1:
            shift = offsetarray[0, 1] - offsetarray[0, 0]
            stream = stream+shift
        else:
            stream = np.polyval(coeff, stream)

    else:
        pass

    # Apply constant offset
    if coffset != None:
        return stream+coffset
    else:
        return stream

def grid_data(x_stream, y_stream, grid):
    """Grid 1d data
    
        Parameters
        ----------
        x_stream : array
            Specify the x data to act on 
        y_stream : array
            Specify the y data to act on
        grid : list, len 3
            Specify start value, end value, and delta

        Returns
        -------
        new_x: numpy array
            equally spaced x scale 
        new_y: numpy array
            y data interpolated on new scale
    """

    # Get min/max values
    xmin = grid[0]
    xmax = grid[1]

    # Calculate number of data points
    numPoints = int(np.ceil((xmax-xmin)/grid[2])) + 1
    
    # Create linear space
    new_x = np.linspace(xmin, xmax, numPoints)

    # Do the interpolation step
    f = interp1d(x_stream, y_stream, fill_value='extrapolate')
    new_y = f(new_x)

    return new_x, new_y

def grid_data2d(x_data, y_data, detector, grid_x=[None, None, None],grid_y=[None, None,None]):
    """Internal function to apply specified grid or ensure otherwise that axes are evenly spaced as this is required to plot an image.
    
        Parameters
        ----------
        x_data: numpy array
        y_data: numpy array
        detector: numpy array
        kwargs:
            grid_x: list
                [start,stop,delta]
            grid_y: list
                [start,stop,delta]

        Returns
        -------
        xmin: float
        xmax: float
        ymin: float
        ymax: float
        new_x: numpy array
        new_y: numpy array
        new_z: numpy array
    """

    # Do auto-grid if not specified otherwise
    # Take step-size as smallest delta observed in data array
    if grid_x == [None, None, None]:
        xmin = x_data.min()
        xmax = x_data.max()
        x_points = int(
            np.ceil((xmax-xmin)/np.abs(np.diff(x_data)).min())) + 1

    else:
        xmin = grid_x[0]
        xmax = grid_x[1]
        x_points = int(np.ceil((xmax-xmin)/grid_x[2])) + 1

    # Same as above, now for second axis.
    if grid_y == [None, None, None]:
        ymin = y_data.min()
        ymax = y_data.max()
        y_points = int(
            np.ceil((ymax-ymin)/np.abs(np.diff(y_data)).min())) + 1

    else:
        ymin = grid_y[0]
        ymax = grid_y[1]
        y_points = int(np.ceil((ymax-ymin)/grid_y[2])) + 1

    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    # Limit matrix to 13107200 entries
    elements = x_points*y_points
    if elements > 13107200:
        # Reduce matrix equally among all dimensions
        #norm = int(np.ceil(np.sqrt(elements/13107200)))
        norm = int(np.ceil(elements/13107200)) # only reduce x
        x_points = int(x_points/norm)
        #y_points = int(y_points/norm)
        warnings.warn(f"Reduced grid size by factor {norm} to maintain memory allocation less than 100MB.")

    # Interpolate the data with given grid
    f = interp2d(x_data, y_data, np.transpose(detector))

    new_x = np.linspace(xmin, xmax, x_points, endpoint=True)
    new_y = np.linspace(ymin, ymax, y_points, endpoint=True)

    # Evaluate image on evenly-spaced grid
    new_z = f(new_x, new_y)

    return xmin, xmax, ymin, ymax, new_x, new_y, new_z
    

def grid_data_mesh(x_data,y_data,z_data):
    """Internal function to generate scatter histogram for 3 independent SCA streams.
    
        Parameters
        ----------
        x_data: numpy array
        y_data: numpy array
        z_data: numpy array
    
        Returns
        -------
        xmin: float
        xmax: float
        ymin: float
        ymax: float
        xedge: numpy array
            xedges as returned from numpy histogram2d
        yedge: numpy array
            yedges as returned from numpy histogram2d
        new_z: numpy array
            2d matrix data
        zmin: float
        zmax: float
    """

    # Get the min/max data
    xmin = x_data.min()
    xmax = x_data.max()
    ymin = y_data.min()
    ymax = y_data.max()
    zmin = z_data.min()
    zmax = z_data.max()

    # Sort out the unique values on scales
    # Need this to generate histogram bins
    xunique = np.unique(x_data)
    yunique = np.unique(y_data)

    # Determine the number of bins
    xbin = len(xunique)
    ybin = len(yunique)

    # Limit array size to 100MB (=104857600 bytes)
    # Numpy float64 array element requires 8 bytes
    # Limit matrix to 13107200 entries
    elements = xbin*ybin
    if elements > 13107200:
        # Reduce matrix equally among all dimensions
        norm = int(np.ceil(np.sqrt(elements/13107200)))
        xbin = int(xbin/norm)
        ybin = int(ybin/norm)
        warnings.warn(f"Reduced grid size by factor {norm} to maintain memory allocation less than 100MB.")

    # Calculate histogram
    new_z, xedge, yedge = np.histogram2d(x_data, y_data, bins=[xbin, ybin], range=[
                                            [xmin, xmax], [ymin, ymax]], weights=z_data)
    
    # Need to transpose data, to maintain compatibility with regular matrix notation
    new_z = np.transpose(new_z)
    new_x = np.linspace(xedge.min(),xedge.max(),len(xedge)-1)
    new_y = np.linspace(yedge.min(),yedge.max(),len(yedge)-1)

    return xmin, xmax, ymin, ymax, new_x, new_y, new_z, zmin, zmax

def bin_data(x_data,y_data,binsize):
    """Reduce noise by averaging data points via binning mechanisms
    
        Parameters
        ----------
        x_data : array
            Specify the x data to act on
        y_data : array
            Specify the y data to act on
        binsize : int
            Specify how many data points to combine
            Must be exponents of 2

        Returns
        -------
        new_x: numpy array
            Mean of the data in bins
        new_y: numpy array
            y-values for the bins
    """

    # Caluclate how many bins
    bins = len(x_data)/binsize

    # Split the data into the bins
    try:
        x_splits = np.split(x_data,bins)
        y_splits = np.split(y_data,bins)
    except Exception as e:
        warnings.warn("Could not split specified quantity in equally split subarrays. Adjust the bin size.")
        raise Exception(e)

    new_x = list()
    new_y = list()

    # Calculate the mean for all x and y values, respectively, in the bin
    for idx,val in enumerate(x_splits):
        new_x.append(np.mean(val))
        new_y.append(np.mean(y_splits[idx]))

    return np.array(new_x), np.array(new_y)

def apply_savgol(x,y,window,polyorder,deriv):
    """Appply smoothing and take derivatives
    
        Parameters
        ----------
        x : array
            x data
        y : array
            y data
        window : int
            Length of the moving window for the Savitzgy-Golay filter
        polyorder : int
            Order of the fitted polynomial
        deriv : int
            Order of the derivative
            Choose "0" if only smoothing requested

        Returns
        -------
        new_x: numpy array
        smooth_y: numpy array
    """

    xmin = x.min()
    xmax = x.max()

    # Caluclate minimum distance between data points to evaluate on evenly spaced grid
    x_diff = np.abs(np.diff(x)).min()
    new_x, new_y  = grid_data(x,y,[xmin,xmax,x_diff])

    # Set default parameters as per savgol docs
    if deriv == 0:
        delta = 1
    else:
        delta = x_diff
    smooth_y = savgol_filter(new_y,window,polyorder,deriv,delta)

    return new_x,smooth_y