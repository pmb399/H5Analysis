"""Collection of simple math operations that may be applied to data"""

# Scientific modules
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from .interpolate import interp1d, interp2d
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
        deg = len(offsetarray)-1
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

#########################################################################################
    
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
    new_y = interp1d(x_stream, y_stream, new_x)

    return new_x, new_y

#########################################################################################

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

    new_x = np.linspace(xmin, xmax, x_points, endpoint=True)
    new_y = np.linspace(ymin, ymax, y_points, endpoint=True)

    # Evaluate image on evenly-spaced grid and interpolate
    new_z = interp2d(x_data, y_data, detector, new_x, new_y)

    return new_x, new_y, new_z
    
#########################################################################################

def grid_data_mesh(x_data,y_data,z_data,binsize_x,binsize_y,bins=None):
    """Internal function to generate scatter histogram for 3 independent SCA streams.
    
        Parameters
        ----------
        x_data: numpy array
        y_data: numpy array
        z_data: numpy array
        binsize_x: int
            puts x-data in bins of specified size
        binsize_y: int
            puts y-data in bins of specified size
        kwargs:
            bins: tuple
                Set the number of bins in the (x-direction,y-direction) explicitly
    
        Returns
        -------
        xedge: numpy array
            xedges as returned from numpy histogram2d
        yedge: numpy array
            yedges as returned from numpy histogram2d
        new_z: numpy array
            2d matrix data
    """

    # Get the min/max data
    xmin = x_data.min()
    xmax = x_data.max()
    ymin = y_data.min()
    ymax = y_data.max()

    # Need to know the number of unique steps in each direction
    # Need this to generate histogram bins
    # Problem: What if np.unique does not work because we are using Encoder Feedback
    # Answer: Base the number of bins off the largest difference in arrays and back-calculate second scale
    if np.abs(np.diff(x_data)).max() > np.abs(np.diff(y_data)).max():
        ybin = len(np.where(np.abs(np.diff(x_data))>0.8*np.abs(np.diff(x_data)).max())[0])+1
        xbin = int(len(x_data)/ybin)
    else:
        xbin = len(np.where(np.abs(np.diff(y_data))>0.8*np.abs(np.diff(y_data)).max())[0])+1
        ybin = int(len(x_data)/xbin)

    if bins!=None and isinstance(bins,tuple):
        xbin = bins[0]
        ybin = bins[1]

    # Problem with Encoder Feedback: Could have empty bins
    # Solution: Interpolate data on evenly spaced grid, take out non-existent data
    # Then, determine final bin sizes 
    interm_x = np.linspace(xmin,xmax,xbin)
    interm_y = np.linspace(ymin,ymax,ybin)

    # Get the x,y mesh grid with unique data points
    x,y = np.meshgrid(np.round(interm_x,decimals=4),np.round(interm_y,decimals=4))
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Interpolate z-data on new grid and drop NaN data points
    big_z = LinearNDInterpolator(list(zip(x_data,y_data)),z_data)(x_flat,y_flat)
    drop = ~np.isnan(big_z)
    dropped_z = big_z[drop]
    a = x_flat[drop]
    b = y_flat[drop]

    # Determine unique points in dropped grid
    xunique = np.unique(a)
    yunique = np.unique(b)

    # Determine the number of bins from unique points in dropped grid
    xbin = len(xunique)
    ybin = len(yunique)

    # Apply the binning if requested
    if isinstance(binsize_x,int):
        xbin = int(np.ceil(xbin/binsize_x))
    if isinstance(binsize_y,int):
        ybin = int(np.ceil(ybin/binsize_y))
        
    ### ### ### ### ### ### ###

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
    new_z, xedge, yedge = np.histogram2d(a, b, bins=[xbin, ybin], range=[
                                            [a.min(), a.max()], [b.min(), b.max()]], weights=dropped_z)
    
    # Need to transpose data, to maintain compatibility with regular matrix notation
    new_z = np.transpose(new_z)
    new_x = np.linspace(xedge.min(),xedge.max(),len(xedge)-1)
    new_y = np.linspace(yedge.min(),yedge.max(),len(yedge)-1)

    return new_x, new_y, new_z

#########################################################################################

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

    return bin_shape_1d(x_data,binsize), bin_shape_1d(y_data,binsize)

def bin_shape_1d(arr,width):
    """ Bins 1d arrays by truncating the array at the end if necessary and calculating the mean in each bin

        Parameters
        ----------
        arr: numpy array
            1d array with data to be binned
        width: int
            Binsize, how many data points per bin
            
        Returns
        -------
        arr: numpy array
            Binned 1d data
    """
    return np.mean(np.split(arr[0:int(np.floor(np.shape(arr)[0]/width)*width)],int(np.floor(np.shape(arr)[0]/width))),axis=1)

def bin_shape_x(arr,width):
    """ Bins 2d matrices in the x direction, i.e. it maintains the number of rows, by truncating the matrix at the end if necessary and calculating the mean in each bin

        Parameters
        ----------
        arr: numpy array
            2d array with data to be binned
        width: int
            Binsize, how many data points per bin
            
        Returns
        -------
        arr: numpy array
            Binned 2d data
    """
    return arr[:,0:int(np.floor(np.shape(arr)[1]/width)*width)].reshape(-1,width).mean(axis=1).reshape(np.shape(arr)[0],int(np.floor(np.shape(arr)[1]/width)))

def bin_shape_y(arr,width):
    """ Bins 2d matrices in the y direction, i.e. it maintains the number of columns, by truncating the matrix at the end if necessary and calculating the mean in each bin

        Parameters
        ----------
        arr: numpy array
            2d array with data to be binned
        width: int
            Binsize, how many data points per bin
            
        Returns
        -------
        arr: numpy array
            Binned 2d data
    """
    return np.transpose(np.transpose(arr[0:int(np.floor(np.shape(arr)[0]/width)*width),:]).reshape(-1,width).mean(axis=1).reshape(np.shape(arr)[1],int(np.floor(np.shape(arr)[0]/width))))

#########################################################################################

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

#########################################################################################

def handle_eval(expr,vars):
    """ Replaces any NaN and inf values from evaluated array 
    
        Parameters
        ----------
        expr: string
            expression to be evaluated
        vars: dict
            dictionary of local variables need to perform evaluation

        Returns
        -------
        numpy array

    """

    # Math functions to handle eval
    from numpy import log as ln
    from numpy import log10 as log
    from numpy import exp
    from numpy import max, min

    # Update the local variables to make everything available needed to perform eval
    locals().update(vars)
    # Eval the expression and replace NaN and inf entries with 0
    return np.nan_to_num(eval(expr),nan=0,posinf=0,neginf=0)