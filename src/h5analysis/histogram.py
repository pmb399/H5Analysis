"""Processing of (x,y,z) scatter data as histogram"""

# Scientific modules
import numpy as np

# Data reader
from .ReadData import Data

# Data parser
from .parser import parse
from numpy import log as ln
from numpy import log10 as log
from numpy import exp
from numpy import max, min

# Utilities
from .datautil import get_roi, get_indices, mca_roi, strip_roi, stack_roi
from .util import check_key_in_dict

# Simple math OPs
from .simplemath import grid_data_mesh, apply_offset, handle_eval

def load_histogram(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=None, binsize_y=None):
    """ Internal function to generate scatter plots for (x,y,z) SCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            key name or alias
        y_stream: string
            key name or alias
        z_stream: string
            key name or alias
        *args: ints
            scan numbers, comma separated
        kwargs:
            norm: boolean
                normalizes to [0,1]
            xoffset: list
                fitting offset (x-stream)
            xcoffset: float
                constant offset (x-stream)
            yoffset: list
                fitting offset (y-stream)
            ycoffset: float
                constant offset (y-stream)
            binsize_x: int
                puts x-data in bins of specified size
            binsize_y: int
                puts y-data in bins of specified size

        Returns
        -------
        data: dict
        """

    # Generate dictionary to store data
    data = dict()

    # Iterate over all scans
    for arg in args:

        # reqs: Store names of the requested data streams
        # rois: rois[stream][reqs]['req'/'roi']
        reqs = list()
        rois = dict()

        # Create h5 Data object
        data[arg] = Data(config,file,arg)
        data[arg].scan = arg

        # Analyse stream requests with parser
        # Get lists of requisitions
        contrib_x_stream = parse(x_stream)
        contrib_y_stream = parse(y_stream)
        contrib_z_stream = parse(z_stream)

        # Strip the requsitions and sort reqs and rois
        reqs, rois = strip_roi(contrib_x_stream,'x', reqs, rois, config)
        reqs, rois = strip_roi(contrib_y_stream,'y', reqs, rois, config)
        reqs, rois = strip_roi(contrib_z_stream,'z', reqs, rois, config)

        # Get the data for all reqs
        all_data = data[arg].Scan(reqs)

        # Set up a stream_convert in which we will replace the strings with local data variables
        # and evaluate the expression later
        x_stream_convert = x_stream
        y_stream_convert = y_stream
        z_stream_convert = z_stream

        # Get the 1d data
        data[arg].x_data = get_hist_stream(contrib_x_stream,x_stream_convert,'x',arg,rois,all_data)
        data[arg].y_data = get_hist_stream(contrib_y_stream,y_stream_convert,'y',arg,rois,all_data)
        data[arg].z_data = get_hist_stream(contrib_z_stream,z_stream_convert,'z',arg,rois,all_data)

        # Apply offsets
        data[arg].x_data = apply_offset(data[arg].x_data, xoffset, xcoffset)
        data[arg].y_data = apply_offset(data[arg].y_data, yoffset, ycoffset)

        # Normalize if requested
        if norm == True:
            data[arg].z_data = data[arg].z_data / np.max(data[arg].z_data)

        # Do some error checking to ensure matching dimensions
        if len(data[arg].x_data) != len(data[arg].y_data) or len(data[arg].y_data) != len(data[arg].z_data):
            raise Exception("Error in x-y-z stream lengths.")

        # Calculate the 2d histogram
        xedge, yedge, new_z = grid_data_mesh(data[arg].x_data,data[arg].y_data,data[arg].z_data,binsize_x,binsize_y)
        data[arg].new_x = xedge
        data[arg].new_y = yedge
        data[arg].new_z = new_z
        data[arg].xlabel = x_stream
        data[arg].ylabel = y_stream
        data[arg].zlabel = z_stream
        data[arg].filename = file

    return data

def get_hist_stream(contrib_stream,convert,stream,arg,rois,all_data):
    """ Get and evaluate each stream requests
    
        Parameters
        ----------
        contrib_stream: list
            all contributions to stream
        convert: string
            stream convert, copy
        stream: string
            name of the string
        arg: int
            current scan
        rois: dict
        all_data: dict
        
        Returns
        -------
        data: numpy array
            evaluated
    
    """

    for i,contrib in enumerate(contrib_stream):
        # Check if component has ROI
        if check_key_in_dict(contrib,rois[stream]):
             # Check that dim = 2
            if len(np.shape(all_data[rois[stream][contrib]['req']])) == 2:
                # Check correct ROI type
                if isinstance(rois[stream][contrib]['roi'],tuple):
                    # Get indices and reduce data
                    low,high = get_indices(rois[stream][contrib]['roi'],all_data[f"{rois[stream][contrib]['req']}_scale"])
                    data = mca_roi(all_data[rois[stream][contrib]['req']],low,high,1,scale=all_data[f"{rois[stream][contrib]['req']}_scale"])

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_{stream}"] = data
                    convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")
                else:
                    raise Exception(f"Error in specified ROI {rois[stream][contrib]['roi']} for {contrib}")

             # Check that dim = 3
            elif len(np.shape(all_data[rois[stream][contrib]['req']])) == 3:
                # Check correct ROI type
                if isinstance(rois[stream][contrib]['roi'],dict):
                    # Get indices and reduce data
                    scale1 = np.average(all_data[f"{rois[stream][contrib]['req']}_scale1"],axis=0)
                    scale2 = np.average(all_data[f"{rois[stream][contrib]['req']}_scale2"],axis=0)
                    idxLow1,idxHigh1 = get_indices(rois[stream][contrib]['roi']['roi_list'][0],scale1)
                    idxLow2,idxHigh2 = get_indices(rois[stream][contrib]['roi']['roi_list'][1],scale2)

                    data = stack_roi(all_data[f"{rois[stream][contrib]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,rois[stream][contrib]['roi']['roi_axes'],scale1=scale1,scale2=scale2)

                    # Check correct data dimensions of reduced data
                    if len(np.shape(data)) == 1:
                        # Add data to locals
                        locals()[f"s{arg}_val{i}_{stream}"] = data
                        convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")
                    else:
                        raise Exception(f'Data dimensionality ({contrib}) incompatible with loader. Check integration axes.')

                else:
                    raise Exception(f"Error in specified ROI {rois[stream][contrib]['roi']['roi_list']} for {contrib}")
            else:
                raise Exception(f"Wrong {stream} dimensions")

        # No ROI is specified
        else:
            # Check correct data dimensions
            if len(np.shape(all_data[contrib])) == 1:
                # Add data to locals
                locals()[f"s{arg}_val{i}_{stream}"] = all_data[contrib]
                convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")

            # Reduce among all scale dimensions if dim=2 or dim=3
            elif len(np.shape(all_data[contrib])) == 2:
                # Apply automatic reduction
                data = mca_roi(all_data[contrib],None,None,1,scale=all_data[f"{contrib}_scale"])

                # Add data to locals
                locals()[f"s{arg}_val{i}_{stream}"] = data
                convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")

            elif len(np.shape(all_data[contrib])) == 3:
                # Apply automatic 
                scale1 = np.average(all_data[f"{contrib}_scale1"],axis=0)
                scale2 = np.average(all_data[f"{contrib}_scale2"],axis=0)
                data = stack_roi(all_data[contrib],None,None,None,None,None,None,(1,2),scale1=scale1,scale2=scale2)

                # Add data to locals
                locals()[f"s{arg}_val{i}_{stream}"] = data
                convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")

            else:
                raise Exception(f"Wrong input dimension {contrib}")
                
    return handle_eval(convert,locals())