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

def get_roi(roi):
    """Gets the roi when ':' separated
    
        Parameters
        ----------
        roi: string

        Returns:
        -------
        roi: tuple, dict
            tuple for 2d data (low,high)
            dict for 3d data
                dict['roi_list] = list with tuples
                dict['roi_axes] = tuple of integration axes
    
    """

    ## Helper function
    def split_roi(roi):
        """Helper function

            Parameters
            ----------
            roi: single roi string

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
                except:
                    if roi.split(":")[0] == 'None':
                        roi_low = None
                    else:
                        raise Exception('Error with lower bound in ROI')
                try:
                    roi_high = float(roi.split(":")[1])
                except:
                    if roi.split(":")[1] == 'None':
                        roi_high = None
                    else:
                        raise Exception("Error with upper bound in ROI")

            # Assume only one value, use as upper and lower limit   
            else:
                if roi == 'None':
                    roi_low = None
                    roi_high = None
                else:
                    roi_low = float(roi)
                    roi_high = float(roi)
                
            return (roi_low, roi_high)
        
        except:
            warnings.warn("Did not understand ROI type")
            raise Exception("Did not understand ROI type")
    
    # Check whether 1 or multiple comma separated ROIs are defined
    if not ',' in roi:
        # This is for the 2d MCA case
        # Store all ROI information as tuple
        roi_clean = split_roi(roi)
    else:
        # This is for the 3d STACK case
        
        # Keep track of sum axes
        sum_axes = list()
        roi_list = list()

        # Go through individual ROIs separately
        for i,roi_tup in enumerate(roi.split(',')):
            try: # Check if sums specified with curely braces
                search = re.search('\{(.*)\}', roi_tup)

                # Remove curely braces and get regular ROI
                roi_list.append(split_roi(search.group(1)))
                # Append those ROIs with curely braces to sum axes
                sum_axes.append(i+1) # start iterating with index 1 since double indices need to catch axes 1 and 2 of stack, not independent axis 0

            except Exception as e:
                roi_list.append(split_roi(roi_tup))

        # Convert sum_axes list to tuple for direct feed to np.sum later
        sum_axes = tuple(sum_axes)
        
        # Store all ROI information in dict
        roi_clean = dict()
        roi_clean['roi_list'] = roi_list
        roi_clean['roi_axes'] = sum_axes

    # Return tuple/dict
    return roi_clean

#########################################################################################

def strip_roi(contrib_reqs,stream, reqs, rois):
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
        try:
            # Split the ROI at"[" and "]"
            # Then get the appropriate ROI (tuple or dict)
            strip = reqroi.split("[")[1].rstrip("]")
            roi = get_roi(strip)

            # Append the split off req name and append
            req = reqroi.split("[")[0]
            reqs.append(req)

            # Store all ROI information in contribution dict
            # Add contribution dict to stream dict
            req_roi['req'] = req
            req_roi['roi'] = roi
            stream_rois[reqroi] = req_roi

        except:
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
        ---------
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