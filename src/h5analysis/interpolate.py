""" Replacement of the scipy interp1 and interp2 functions applied consistently throughout this project"""
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def interp1d(x,y,new_x,**kwargs):
    """ Replaces the legacy scipy.interpolate.interp1d function (see here: https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection) with numpy.interp.

        Parameters
        ----------
        x: 1d array
            x data
        y: 1d array
            y data
        new_x: 1d array
            x data for interpolation
        ** kwargs:
            as per numpy.interp
    """

    # Set the boundaries
    kwargs.setdefault("left",np.nan)
    kwargs.setdefault("right",np.nan)

    # Check that the x values are strictly increasing
    # Otherwise sort values
    if np.all(np.diff(x) > 0):
        return np.interp(new_x,x,y,**kwargs)
    else:
        xarg = np.argsort(x)
        x = x[xarg]
        y = np.take(y,xarg,axis=0)
        return np.interp(new_x,x,y,**kwargs)

def interp2d(x,y,z,new_x,new_y,**kwargs):
    """ Replaces the legacy scipy.interpolate.interp2d function (see here: https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff) with RegularGridInterpolator.

        Parameters
        ----------
        x: 1d array
            x data
        y: 1d array
            y data
        z: 2d array
            z data
        new_x: 1d array
            x data for interpolation
        new_y: 1d array
            y data for interpolation
        ** kwargs:
            as per scipy.interpolate.RegularGridInterpolator
    """
    
    # Override default behaviour when interpolating out of bounds
    # to ensure compatibility with legacy scipy.interpolate.interp2d 
    kwargs.setdefault("bounds_error",False)

    # Generate meshgrid to evaluate interpolation
    xx,yy = np.meshgrid(new_x,new_y,indexing='ij',sparse=True)
    
    # Try to run RegularGridInterpolator
    # Requires (x,y) to be strictly ascending
    # If this check fails, sort the x and y arrays, and order z accordingly
    try:
        return RegularGridInterpolator((x,y),z.T,**kwargs)((xx,yy)).T
    except:
        xarg = np.argsort(x)
        yarg = np.argsort(y)
        x = x[xarg]
        y = y[yarg]
        z = np.take(z,yarg,axis=0)
        z = np.take(z,xarg,axis=1)
        return RegularGridInterpolator((x,y),z.T,**kwargs)((xx,yy)).T