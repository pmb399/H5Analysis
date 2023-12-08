import numpy as np
from .util import get_roi, check_idx

def detector_norm(detector_mca,norm_by):
    """Normalize the detector data by mesh current."""
    return np.true_divide(detector_mca,norm_by[:, None])
    
def mca_roi(mca,roi_scale,roi,axis):
    if isinstance(roi,tuple):
        idx_start = (np.abs(roi[0] - roi_scale)).argmin()
        idx_end = (np.abs(roi[1] - roi_scale)).argmin()

        idx_start, idx_end = check_idx(idx_start,idx_end)
        if axis == 0:
            return mca[idx_start:idx_end, :].sum(axis=axis)
        elif axis == 1:
            return mca[:,idx_start:idx_end].sum(axis=axis)
        else:
            raise UserWarning('Wrong axis.')
    
    else:
        return np.sum(mca, axis=axis)
    
def subtract_background(mca,bg):
    if bg.ndim > 1:
            # 1 Sum the background
            background_sum = bg.sum(axis=0)

            # 2 Normalize to average background frame
            background_spec = np.true_divide(background_sum, len(bg))

            # 3 Subtract average bg frame from all frames
            mca = np.subtract(mca, background_spec)
      
            return mca
    else:
        raise Exception("Wrong background specified.")