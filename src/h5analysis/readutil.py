"""Functions for data processing during read-in"""

# Scientific modules
import numpy as np

def detector_norm(detector_mca,norm_by):
    """Normalize the detector data by mesh current.
    
        Parameters
        ----------
        detector_mca: numpy array
            2d data
        norm_by: numpy array
            1d data, with length equal to first axis of detector_mca

        Returns
        -------
        Division of detector_mca and norm_by array
    """

    return np.true_divide(detector_mca,norm_by[:, None],where=norm_by[:, None]!=0)

#########################################################################################

def stack_norm(detector_stack,norm_by):
    """Normalize the stack data by mesh current.
    
        Parameters
        ----------
        detector_stack: numpy array
            2d data
        norm_by: numpy array
            1d data, with length equal to first axis of detector_stack

        Returns
        -------
        Division of detector_mca and norm_by array

    """
    
    return np.true_divide(detector_stack,norm_by[:, None, None],where=norm_by[:, None, None]!=0)