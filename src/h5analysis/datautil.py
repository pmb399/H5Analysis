import numpy as np
import warnings
import re

def mca_roi(mca,idx_start,idx_end,sumaxis,scale=None,ind_axis=None):
    if len(np.shape(mca)) != 2:
        raise Exception("Function only defined for dim 2")

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
    
    if sumaxis == 0:
        return mca[idx_start:idx_end, :].sum(axis=sumaxis)
    elif sumaxis == 1:
        return mca[:, idx_start:idx_end].sum(axis=sumaxis)
    else:
        raise UserWarning('Wrong axis.')

#########################################################################################
    
def stack_roi(stack,idxLowI,idxHighI,idxLow1,idxHigh1,idxLow2,idxHigh2,integration_axes,scale1=None,scale2=None,ind_axis=None):
    if len(np.shape(stack)) != 3:
        raise Exception("Function only defined for dim 3")
    
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
        
    return np.sum(stack[idxLowI:idxHighI,idxLow1:idxHigh1,idxLow2:idxHigh2],axis=integration_axes)
    

#########################################################################################

def get_roi(roi):
    """Gets the roi when ':' separated"""
    def split_roi(roi):
        try:
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
                    
            else:
                roi_low = float(roi)
                roi_high = float(roi)
                
            return (roi_low, roi_high)
        
        except:
            warnings.warn("Did not understand ROI type")
            raise Exception("Did not understand ROI type")
    
    if not ',' in roi:
        roi_clean = split_roi(roi)
    else:
        sum_axes = list()
        roi_list = list()
        for i,roi_tup in enumerate(roi.split(',')):
            try: # Check if sums specified
                search = re.search('\{(.*)\}', roi_tup)
                roi_list.append(split_roi(search.group(1)))
                sum_axes.append(i+1) # start iterating with index 1 since double indices need to catch axes 1 and 2 of stack, not independent axis 0
            except Exception as e:
                roi_list.append(split_roi(roi_tup))

        sum_axes = tuple(sum_axes)
        
        roi_clean = dict()
        roi_clean['roi_list'] = roi_list
        roi_clean['roi_axes'] = sum_axes

    return roi_clean

#########################################################################################

def strip_roi(contrib_reqs,stream, reqs, rois):
        stream_rois = dict()
        for reqroi in contrib_reqs:
            req_roi = dict()
            try:
                strip = reqroi.split("[")[1].rstrip("]")
                roi = get_roi(strip)
                req = reqroi.split("[")[0]

                reqs.append(req)
                req_roi['req'] = req
                req_roi['roi'] = roi
                stream_rois[reqroi] = req_roi

            except:
                reqs.append(reqroi)
        
        rois[stream] = stream_rois

        return reqs, rois

#########################################################################################

def check_idx(idx_low, idx_high):
    """Check the index of an array. Add +1 to allow slicing."""
    if idx_low != None and idx_high != None:
        if idx_low == idx_high:
            idx_high = idx_low+1

    return idx_low, idx_high

#########################################################################################

def get_indices(roi,roi_scale):
    if isinstance(roi,tuple):
        if not isinstance(roi[0],type(None)):
            idx_start = (np.abs(roi[0] - roi_scale)).argmin()
        else:
            idx_start = None
        if not isinstance(roi[1],type(None)):
            idx_end = (np.abs(roi[1] - roi_scale)).argmin()
        else:
            idx_end = None

        return check_idx(idx_start,idx_end)
    
    elif isinstance(roi,type(None)):
        return None
    
    else:
        raise Exception('No ROI specified')