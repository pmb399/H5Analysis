import numpy as np
from .ReadData import REIXS
from .simplemath import apply_offset, grid_data2d
from .parser import math_stream
from .util import check_key_in_dict
from .spec_config import get_REIXSconfig
from .readutil import detector_norm

def loadSTACKscans(file, stack, arg):
    """Internal function to load STACK data
    
        Parameters
        ----------
        
    """
        
    # Place all loaded REIXS objects in data dictionary
    data = dict()

    data[arg] = REIXSobj = REIXS(file,arg)
    data[arg].scan = arg

    kwargs = dict()
    data[arg].Scan(stack,None,kwargs)

    data[arg].x_min = data[arg].data_scale.min()
    data[arg].x_max = data[arg].data_scale.max()
    data[arg].y_min = data[arg].image_scale.min()
    data[arg].y_max = data[arg].image_scale.max()

    return data
