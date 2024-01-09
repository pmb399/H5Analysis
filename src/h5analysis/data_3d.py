# Scientific modules
import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp
from numpy import max, min

# Data reader
from .ReadData import Data

# Data utilities
from .parser import parse
from .util import check_key_in_dict
from .simplemath import apply_offset, grid_data2d
from .datautil import strip_roi, get_indices, mca_roi, stack_roi
from .readutil import stack_norm

def load_3d(config, file, stack, arg, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None):
    """ Internal function to load STACK data
    
        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            filename
        stack: string
            alias of an image STACK
        args: int
            scan number
        kwargs
            xoffset: list of tuples
                fitted offset (x-stream)
            xcoffset: float
                constant offset (x-stream)
            yoffset: list of tuples
                fitted offset (y-stream)
            ycoffset: float
                constant offset (y-stream)
            grid_x: list
                grid equally spaced in x with [start, stop, delta]
            grid_y: list
                grid equally spaced in y with [start, stop, delta]
            norm_by: string
                norm MCA by defined h5 key or SCA alias

        Returns
        -------
        data: dict
    """
    
    # Place all loaded Data objects in dictionary
    data = dict()

    # Create h5 Data object
    data[arg] = Data(config,file,arg)
    data[arg].scan = arg

    # reqs: Store names of the requested data streams
    # rois: rois[stream][reqs]['req'/'roi']
    reqs = list()
    rois = dict()

    # Analyse x-stream and y-stream requests with parser
    # Get lists of requisitions
    contrib_stack = parse(stack)

    # Strip the requsitions and sort reqs and rois
    reqs, rois = strip_roi(contrib_stack,'s', reqs, rois)

    # Get the data for specified STACK
    all_data = data[arg].Scan(reqs)

    # Set up an stream_convert in which we will replace the strings with local data variables
    # and evaluate the expression later
    s_stream_convert = stack

    has_stack = False

    for i,s in enumerate(contrib_stack):
            # This is only to perform additional math operations
        if check_key_in_dict(s,rois['s']):
            if len(np.shape(all_data[rois]['s'][s]['req'])) == 2:
                low,high = get_indices(rois['s'][s]['roi'],all_data[f"{rois['s'][s]['req']}_scale"])
                data = mca_roi(all_data[rois['s'][s]['req']],low,high,1,scale=all_data[f"{rois['s'][s]['req']}_scale"])

                # Add data to locals
                locals()[f"s{arg}_val{i}_s"] = data[:,None,None]
                s_stream_convert = s_stream_convert.replace(s,f"s{arg}_val{i}_s")

            elif len(np.shape(all_data[rois]['s'][s]['req'])) == 3:
                if isinstance(rois['s'][s]['roi'],dict):
                    # Get ROI indices
                    idxLow1,idxHigh1 = get_indices(rois['s'][s]['roi']['roi_list'][0],all_data[f"{rois['s'][s]['req']}_scale1"])
                    idxLow2,idxHigh2 = get_indices(rois['s'][s]['roi']['roi_list'][1],all_data[f"{rois['s'][s]['req']}_scale2"])

                    data = stack_roi(all_data[f"{rois['s'][s]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,(1,2),scale1=all_data[f"{rois['s'][s]['req']}_scale1"],scale2=all_data[f"{rois['s'][s]['req']}_scale2"])

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_s"] = data[:,None,None]
                    s_stream_convert = s_stream_convert.replace(s,f"s{arg}_val{i}_s")

                else:
                    raise Exception(f"Error in defined ROI {rois['s'][s]['roi']['roi_list']} for {s}")

            else:
                raise Exception(f"Cannot perform math on defined stream {s}")
            
        else:
            if len(np.shape(all_data[s])) == 3:
                if not check_key_in_dict(s,config.h5dict):
                    raise Exception(f"Data Stream {s} must be configured via config dict.")
                
                if not config.h5dict[s]['type'] == "STACK":
                    raise Exception(f"Need to specify an image stack. Error caused by: {s}")
                
                if has_stack == False:
                    # Apply offset
                    x_data = apply_offset(all_data[f"{s}_scale1"], xoffset, xcoffset)
                    y_data = apply_offset(all_data[f"{s}_scale2"], yoffset, ycoffset)

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_s"] = all_data[s]
                    s_stream_convert = s_stream_convert.replace(s,f"s{arg}_val{i}_s")

                    has_stack = True

                else:
                    raise Exception('Can only specify one stack')

            elif len(np.shape(all_data[s])) == 1:
                # This is for additional math ops only
                # Add data to locals
                locals()[f"s{arg}_val{i}_s"] = all_data[s][:,None,None]
                s_stream_convert = s_stream_convert.replace(s,f"s{arg}_val{i}_s")

            else:
                raise Exception(f'Data dimension of {s} unsupported.')
            
    if has_stack == False:
        raise Exception("No stack specified.")
    
    stack_data = eval(s_stream_convert)

    # Normalize MCA data by SCA
    if not isinstance(norm_by,type(None)):
        norm_data = data[arg].Scan(norm_by)
        normalization = norm_data[norm_by]
        my_stack = stack_norm(stack_data,normalization)
    else:
        my_stack = stack_data

    # Iterate over independent axis and grid all data to images
    stack_grid = list()
    for i,img in enumerate(my_stack):
        xmin, xmax, ymin, ymax, new_x, new_y, new_z = grid_data2d(x_data, y_data, img, grid_x=grid_x,grid_y=grid_y)
        stack_grid.append(new_z)

    # Generate 3d stack from gridded z-data in stack_grid list
    # Store all data in dict
    data[arg].stack = np.stack(tuple(stack_grid))
    data[arg].x_min = xmin
    data[arg].x_max = xmax
    data[arg].y_min = ymin
    data[arg].y_max = ymax

    return data