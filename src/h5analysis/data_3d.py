"""Processing of 3d data"""

# Scientific modules
import numpy as np

# Data reader
from .ReadData import Data

# Data utilities
from .parser import parse
from .util import check_key_in_dict
from .simplemath import apply_offset, grid_data2d, handle_eval, bin_shape_1d, bin_shape_x, bin_shape_y
from .datautil import strip_roi, get_indices, mca_roi, stack_roi
from .readutil import stack_norm

def load_3d(config, file, ind_stream, stack, arg, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None,binsize_x=None,binsize_y=None):
    """ Internal function to load STACK data
    
        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            filename
        ind_stream: string
            independent stream, corresponding to stack's first dim
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize: int
                puts data in bins of specified size in the vertical direction

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
    reqs, rois = strip_roi(contrib_stack,'s', reqs, rois, config)

    # Add independent stream to reqs
    reqs.append(ind_stream)

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
                    scale1 = np.average(all_data[f"{rois['s'][s]['req']}_scale1"],axis=0)
                    scale2 = np.average(all_data[f"{rois['s'][s]['req']}_scale2"],axis=0)
                    idxLow1,idxHigh1 = get_indices(rois['s'][s]['roi']['roi_list'][0],scale1)
                    idxLow2,idxHigh2 = get_indices(rois['s'][s]['roi']['roi_list'][1],scale2)

                    data = stack_roi(all_data[f"{rois['s'][s]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,(1,2),scale1=scale1,scale2=scale2)

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

                    if config.h5dict[s]['label1'] != None:
                        xlabel = config.h5dict[s]['label1']
                    else:
                        xlabel = f"{s}_Scale1"

                    if config.h5dict[s]['label2'] != None:
                        ylabel = config.h5dict[s]['label2']
                    else:
                        ylabel = f"{s}_Scale2"

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
    
    stack_data = handle_eval(s_stream_convert,locals())

    # Normalize MCA data by SCA
    if not isinstance(norm_by,type(None)):
        norm_data = data[arg].Scan(norm_by)
        normalization = norm_data[norm_by]
        my_stack = stack_norm(stack_data,normalization)
    else:
        my_stack = stack_data

    # Iterate over independent axis and grid all data to images
    stack_grid = list()
    new_x_list = list()
    new_y_list = list()
    for i,img in enumerate(my_stack):
        # Apply the transpose as per data convention
        img = np.transpose(img)
        # Apply the binning
        if isinstance(binsize_x,int):
            x_data_i = bin_shape_1d(x_data[i],binsize_x)
            img = bin_shape_x(img,binsize_x)
        else:
            x_data_i = x_data[i]

        if isinstance(binsize_y,int):
            y_data_i = bin_shape_1d(y_data[i],binsize_y)
            img = bin_shape_y(img,binsize_y)
        else:
            y_data_i = y_data[i]

        new_x, new_y, new_z = grid_data2d(x_data_i, y_data_i, img, grid_x=grid_x,grid_y=grid_y)
        stack_grid.append(new_z)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

    # Generate 3d stack from gridded z-data in stack_grid list
    # Store all data in dict
    data[arg].stack = np.stack(tuple(stack_grid))
    data[arg].ind_stream = all_data[ind_stream]
    data[arg].str_ind_stream = ind_stream
    data[arg].new_x = new_x_list
    data[arg].new_y = new_y_list

    data[arg].xlabel = xlabel
    data[arg].ylabel = ylabel
    data[arg].zlabel = stack

    data[arg].filename = file

    # Check that independent stream dimension is correct
    if len(data[arg].ind_stream) == np.shape(data[arg].stack)[0]:
        pass
    else:
        raise Exception('Dimension mismatch. Check specified independent stream.')

    return data