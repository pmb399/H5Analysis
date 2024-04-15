"""Utility functions for data processing and reduction"""

import numpy as np
import warnings
import re

def mca_roi(mca,idx_start,idx_end,sumaxis,scale=None,ind_axis=None):
    """ Reduce 2d MCA data to 1d

        Parameters
        ----------
        mca: numpy array
            2d data
        idx_start: int
            index
        idx_end: int
            index
        sumaxis: int
            0,1 for summation direction
        scale: numpy array
            scale corresponding to dim 1
        ind_axis: numpy array
            independent axis corresponding to dim 0

        Returns
        -------
        reduced: numpy array
            summed and reduced 1d data
    """

    # Ensure this function is only used for dim 2
    if len(np.shape(mca)) != 2:
        raise Exception("Function only defined for dim 2")

    # Ensure proper integration axis
    if sumaxis>1:
        raise Exception("Integration axis above dimensions")

    # We assume that data is written sequentially
    # Thus, MCA data is expected in shape (n,m) where m corresponds to the length of the scale
    # Only check whether that is the case, and transpose if not

    if not isinstance(scale,type(None)):
        if len(scale) == np.shape(mca)[1]:
            pass

        else:
            warnings.warn("Error determining integration axis, Transposing matrix.")
            mca = np.transpose(mca)

    if not isinstance(ind_axis,type(None)):
        if len(ind_axis) == np.shape(mca)[0]:
            pass

        else:
            warnings.warn("Error determining integration axis, Transposing matrix.")
            mca = np.transpose(mca)
    
    # Apply the slicing and summation
    if sumaxis == 0:
        return mca[idx_start:idx_end, :].sum(axis=sumaxis)
    elif sumaxis == 1:
        return mca[:, idx_start:idx_end].sum(axis=sumaxis)
    else:
        raise UserWarning('Wrong axis.')

#########################################################################################
    
def stack_roi(stack,idxLowI,idxHighI,idxLow1,idxHigh1,idxLow2,idxHigh2,integration_axes,scale1=None,scale2=None,ind_axis=None):
    """ Reduce 3d MCA data to 2d or 1d

        Parameters
        ----------
        stack: numpy array
            3d data
        idxLowI: int
            index low independent axis
        idxHighI: int
            index high independent axis
        idxLow1: int
            index low scale 1
        idxHigh1: int
            index high scale 1
        idxLow2: int
            index low scale 2
        idxHigh2: int
            index high scale 2
        integration_axes: tuple
            0,1,2 for summation direction (max. 2)
        scale1: numpy array
            scale corresponding to dim 1
        scale2: numpy array
            scale corresponding do dim 2
        ind_axis: numpy array
            independent axis corresponding to dim 0

        Returns
        -------
        reduced: numpy array
            summed and reduced 2d/1d data
    """

    # Ensure proper dimensions
    if len(np.shape(stack)) != 3:
        raise Exception("Function only defined for dim 3")
    
    # Ensure STACK and scales match
    # This particulary requires that the independent axis is 0
    if not isinstance(ind_axis,type(None)):
        if len(ind_axis) == np.shape(stack)[0]:
            pass
        else:
            raise Exception("Error determining integration axis, Transposing matrix.")
    
    if not isinstance(scale1,type(None)):
        if len(scale1) == np.shape(stack)[1]:
            pass
        else:
            raise Exception("Error determining integration axis.")

    if not isinstance(scale2,type(None)):
        if len(scale2) == np.shape(stack)[2]:
            pass
        else:
            raise Exception("Error determining integration axis.")
    
    # Apply data reduction
    return np.sum(stack[idxLowI:idxHighI,idxLow1:idxHigh1,idxLow2:idxHigh2],axis=integration_axes)
    

#########################################################################################

def get_roi(roi, config):
    """Gets the roi when ':' separated
    
        Parameters
        ----------
        roi: string
        config: dict
            h5 configuration

        Returns
        -------
        roi: tuple, dict
            tuple for 2d data (low,high) or dict for 3d data
                dict['roi_list] = list with tuples
                dict['roi_axes] = tuple of integration axes
    
    """

    ## Helper function
    def split_roi(roi, config):
        """Helper function

            Parameters
            ----------
            roi: single roi string
            config: dict
                h5 configuration

            Returns
            -------
            roi: tuple
                (roi_low,roi_high)
        """
        try:
            # If colon separated, extract values as float
            # unless None specified
            if ':' in roi:
                try:
                    roi_low = float(roi.split(":")[0])
                except Exception as e:
                    if roi.split(":")[0] == 'None':
                        roi_low = None
                    else:
                        raise Exception(f'Error with lower bound in ROI. {e}')
                try:
                    roi_high = float(roi.split(":")[1])
                except Exception as e:
                    if roi.split(":")[1] == 'None':
                        roi_high = None
                    else:
                        raise Exception(f"Error with upper bound in ROI. {e}")

            # Assume only one value, use as upper and lower limit   
            else:
                if roi == 'None':
                    roi_low = None
                    roi_high = None
                else:
                    try:
                        roi_low = float(roi)
                        roi_high = float(roi)
                    except:
                        try:
                            roi_low = config.roi_dict[roi][0]
                            roi_high = config.roi_dict[roi][1]
                        except Exception as e:
                            warnings.warn("Did not understand ROI type")
                            raise Exception(e)
                
            return (roi_low, roi_high)
        
        except Exception as e:
            warnings.warn("Did not understand ROI type")
            raise Exception(e)
    
    # Check whether 1 or multiple comma separated ROIs are defined
    if not ',' in roi:
        # This is for the 2d MCA case
        # Store all ROI information as tuple
        roi_clean = split_roi(roi,config)
    else:
        # This is for the 3d STACK case
        
        # Keep track of sum axes
        sum_axes = list()
        roi_list = list()

        # Go through individual ROIs separately
        for i,roi_tup in enumerate(roi.split(',')):
            if '{' in roi_tup and '}' in roi_tup:
                # Check if sums specified with curely braces
                search = re.search('\{(.*)\}', roi_tup)

                # Remove curely braces and get regular ROI
                roi_list.append(split_roi(search.group(1),config))
                # Append those ROIs with curely braces to sum axes
                sum_axes.append(i+1) # start iterating with index 1 since double indices need to catch axes 1 and 2 of stack, not independent axis 0

            else:
                roi_list.append(split_roi(roi_tup,config))

        # Convert sum_axes list to tuple for direct feed to np.sum later
        sum_axes = tuple(sum_axes)
        
        # Store all ROI information in dict
        roi_clean = dict()
        roi_clean['roi_list'] = roi_list
        roi_clean['roi_axes'] = sum_axes

    # Return tuple/dict
    return roi_clean

#########################################################################################

def strip_roi(contrib_reqs,stream, reqs, rois, config):
    """ Split ROIS out of strings and store

        Parameters
        ----------
        contrib_reqs: list
            all requests
        stream: string
            add as additional dict key
        reqs: list
            all stripped reqs
        rois: dict
            all info regarding specified ROIs
        config: dict
            h5 configuration

        Returns
        -------
        reqs: list
        rois: dict
    """

    # Set up dict for specific stream since "rois" is global
    stream_rois = dict()

    # Go through all contributions
    for reqroi in contrib_reqs:
        # Add dict for contribution
        req_roi = dict()
        if '[' in reqroi and ']' in reqroi:
            # Split the ROI at "[" and "]"
            # Then get the appropriate ROI (tuple or dict)
            strip = reqroi.split("[")[1].rstrip("]")
            roi = get_roi(strip,config)

            # Append the split off req name and append
            req = reqroi.split("[")[0]
            reqs.append(req)

            # Store all ROI information in contribution dict
            # Add contribution dict to stream dict
            req_roi['req'] = req
            req_roi['roi'] = roi
            stream_rois[reqroi] = req_roi

        else:
            # If no ROI could be extracted
            # Add plain request to reqs list
            reqs.append(reqroi)
    
    # Append stream dict to global dict
    rois[stream] = stream_rois

    return reqs, rois

#########################################################################################

def check_idx(idx_low, idx_high):
    """Check the index of an array. Add +1 to allow slicing.
    
        Parameters
        ----------
        idx_low: int
        idx_high: int

        Returns
        -------
        idx_low: int
        idx_high: int
    """

    # If idx_low and idx_high are both specified and are equal, add +1
    if idx_low != None and idx_high != None:
        if idx_low == idx_high:
            idx_high = idx_low+1

    return idx_low, idx_high

#########################################################################################

def get_indices(roi,roi_scale):
    """ Get the indices from specified ROI and scale

        Parameters
        ----------
        roi: tuple
            single ROI element
        roi_scale: numpy array
            scale to extract closest index from
    """

    # Ensure we are dealing with single ROI
    if isinstance(roi,tuple):
        # Only do this if None is not specified
        # If value, get closest index

        # index low
        if not isinstance(roi[0],type(None)):
            idx_start = (np.abs(roi[0] - roi_scale)).argmin()
        else:
            idx_start = None
        
        # index high
        if not isinstance(roi[1],type(None)):
            idx_end = (np.abs(roi[1] - roi_scale)).argmin()
        else:
            idx_end = None

        # check indices and return
        return check_idx(idx_start,idx_end)
    
    # if single index None specified
    elif isinstance(roi,type(None)):
        return None
    
    else:
        raise Exception('No ROI specified')
    
#########################################################################################

def get_indices_polygon(pairs,scale1,scale2):
    """ Get the indices for specified coordinate pairs

        Parameters
        ----------
        pairs: list of tuple
            corresponds to coordinates
        scale1: numpy array
            scale to extract closest index from
        scale2: numpy array
            scale to extract closest index from
        """

    x_list = list()
    y_list = list()

    for pair in pairs:
        x_list.append((np.abs(pair[0]-scale1)).argmin())
        y_list.append((np.abs(pair[1]-scale2)).argmin())

    return np.column_stack((y_list,x_list))

#########################################################################################
    
def check_dimensions2d(x,y,z):
    """Ensure dimensions of scales are matching image array and image is gridded evenly
    
    Parameters
    ----------
    x: 1d array
    y: 1d array
    z: 2d array with shape (len(y),len(x))

    """

    dim_z = np.shape(z) # in matrix notation
    len_x = len(x)
    len_y = len(y)

    if dim_z[1] != len_x:
        raise Exception('x dimension of image and scale does not match')
    
    if dim_z[0] != len_y:
        raise Exception('y dimension of image and scale does not match')
    
    # Check that we are indeed plotting an image (equal spacing)
    if not np.unique(np.diff(x).round(decimals=8)).size == 1:
        raise Exception('No even grid for x axis')
    if not np.unique(np.diff(y).round(decimals=8)).size == 1:
        raise Exception('No even grid for y axis')
    
#########################################################################################
    
def bokeh_image_boundaries(x,y):
    """Calculates the boundaries and width for the displayed image
    
    Parameters
    ----------
    x: 1d array
    y: 1d array

    Returns
    -------
    plot_x_corner: coordinates of the lower left corner on x
    plot_y_corner: coordinates of the lower left corner on y
    plot_dw: with of the plot (in data coordinates)
    plot_df: height of the plot (in data coordinates)
    """
    
    # Get the scale limits
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    # Calculate bokeh boundaries
    diff_x  = x[1]-x[0]
    diff_y  =y[1]-y[0]
    plot_x_corner = xmin-diff_x/2
    plot_y_corner = ymin-diff_y/2
    plot_dw = xmax-xmin + diff_x
    plot_dh = ymax-ymin + diff_y

    return plot_x_corner,plot_y_corner, plot_dw,plot_dh