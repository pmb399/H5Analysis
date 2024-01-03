import numpy as np

def detector_norm(detector_mca,norm_by):
    """Normalize the detector data by mesh current."""
    return np.true_divide(detector_mca,norm_by[:, None])

def stack_norm(detector_mca,norm_by):
    """Normalize the stack data by mesh current."""
    return np.true_divide(detector_mca,norm_by[:, None, None])