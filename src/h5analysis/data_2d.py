# Scientific modules
import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp
from numpy import max, min

# Data reader
from .ReadData import Data

# Data utils
from .parser import parse
from .datautil import get_roi, get_indices, mca_roi, strip_roi, stack_roi
from .util import check_key_in_dict
from .simplemath import apply_offset, grid_data2d
from .readutil import detector_norm

# Warnings
import warnings

def load_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None,):
    """ Internal function to load 2d MCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            filename
        x_stream: string
            h5 sca key or alias of the x-stream
        detector: string
            alias of the MCA detector
        *args: ints
            scan numbers
        kwargs
            norm: boolean, None
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

    # Store all data in data dict
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
        data[arg].xlabel = x_stream
        data[arg].zlabel = detector
        data[arg].filename = file
        
        # Analyse x-stream and y-stream requests with parser
        # Get lists of requisitions
        contrib_x_stream = parse(x_stream)
        contrib_z_stream = parse(detector)

        # Strip the requsitions and sort reqs and rois
        reqs, rois = strip_roi(contrib_x_stream,'x', reqs, rois)
        reqs, rois = strip_roi(contrib_z_stream,'z', reqs, rois)

        # Get the data for all reqs
        all_data = data[arg].Scan(reqs)

        # x stream
        # Set up an x_stream_convert in which we will replace the strings with local data variables
        # and evaluate the expression later
        x_stream_convert = x_stream

        for i,x in enumerate(contrib_x_stream):

            # Check if x component has ROI
            if check_key_in_dict(x,rois['x']):
                # Check that dim(x) = 1
                if len(np.shape(all_data[rois['x'][x]['req']])) == 1:
                    # Check that we only have 1 ROI x-stream to reduce to dim 0
                    if len(contrib_x_stream) != 1:
                        raise Exception('Only one ROI x-stream supported.')
                    # Ensure ROI is of correct type
                    if isinstance(rois['x'][x]['roi'],tuple):
                        # x dimension will be 0 and we only obtain indices
                        dim = 0
                        xlow,xhigh = get_indices(rois['x'][x]['roi'],all_data[rois['x'][x]['req']])
                    else:
                        raise Exception(f"Error in specified ROI {rois['x'][x]['roi']} for {x}")

                else:
                    raise Exception(f"Wrong x dimensions for {x}")
            
            # If x component has no ROI
            else:
                if len(np.shape(all_data[x])) == 1:
                    # len(contrix_x_stream) requirement above implicitly verifies that
                    # we can only have multiple x components if dim=1
                    dim = 1

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_x"] = all_data[x]
                    x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                else:
                    raise Exception(f"Wrong input dimension: {x}")

        # Check proper dimensions for x-stream
        if not (dim==0 or dim == 1):
            raise Exception('Error defining x-stream')
        
        # If dim_x == 1, can evaluate expression
        if dim == 1:
            # Assign the calculated result to the x_stream of the object in data dict
            data[arg].x_data = eval(x_stream_convert)
                
        # z-stream
        # Set up an z_stream_convert in which we will replace the strings with local data variables
        # and evaluate the expression later
        z_stream_convert = detector

        for i,z in enumerate(contrib_z_stream):
            # Check if requisition has ROIs
            if check_key_in_dict(z,rois['z']):
                # Need to have a STACK-ROI for this, thus require dim(z)=3
                if len(np.shape(all_data[rois['z'][z]['req']])) == 3:
                    # Check that ROI is appropriate
                    if isinstance(rois['z'][z]['roi'],dict):
                        # Get ROI indices
                        scale1 = np.average(all_data[f"{rois['z'][z]['req']}_scale1"],axis=0)
                        scale2 = np.average(all_data[f"{rois['z'][z]['req']}_scale2"],axis=0)
                        idxLow1,idxHigh1 = get_indices(rois['z'][z]['roi']['roi_list'][0],scale1)
                        idxLow2,idxHigh2 = get_indices(rois['z'][z]['roi']['roi_list'][1],scale2)

                        # Reduce STACK once along specified scale axis if dim(x) = 1
                        if dim == 1:
                            # Calculate resulting 2d data
                            z_data = stack_roi(all_data[f"{rois['z'][z]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,rois['z'][z]['roi']['roi_axes'],scale1=scale1,scale2=scale2)
                            
                            # Ensure result is of correct dimensions
                            if len(np.shape(z_data)) == 2:
                                # Add data to locals
                                locals()[f"s{arg}_val{i}_z"] = z_data
                                z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                                # Determine the remaining axis
                                integration_axes = rois['z'][z]['roi']['roi_axes']
                                all_axes = {1,2}
                                y_axis_raw = all_axes - set(integration_axes)
                                if not len(list(y_axis_raw)) == 1: # Can only have exactly one axis as per dimensionality requirements
                                    raise Exception(f'Error determining proper integration axes ({z})')
                                else:
                                    y_axis = list(y_axis_raw)[0] # convert to single element (from set to list, then slice)

                                    # Set the remaining (second) axis as y-data
                                    if y_axis == 1:
                                        data[arg].ylabel = f"{rois['z'][z]['req']}_scale1"
                                        data[arg].y_data = np.average(all_data[f"{rois['z'][z]['req']}_scale1"],axis=0)[idxLow1:idxHigh1]
                                    elif y_axis == 2:
                                        data[arg].ylabel = f"{rois['z'][z]['req']}_scale2"
                                        data[arg].y_data = np.average(all_data[f"{rois['z'][z]['req']}_scale2"],axis=0)[idxLow2:idxHigh2]
                                    else:
                                        raise Exception(f"Wrong axis defined ({z}).")
                                    
                            elif len(np.shape(z_data)) == 1:
                                # This is only to perform additional math operations
                                locals()[f"s{arg}_val{i}_z"] = z_data[:,None]
                                z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                            else:
                                raise Exception(f'Data dimensionality incompatible with loader ({z}). Check integration axes.')
                            
                        # Reduce STACK once along independent axis if dim(x) == 0
                        elif dim == 0:
                            if not isinstance(xlow,type(None)) and not isinstance(xhigh,type(None)):
                                if xlow > xhigh:
                                    warnings.warn("xlow>xhigh.\nEither select a single value or [None:None] to integrate over all.\nThis most likely happens because the chosen x-stream is not monotonic.")
                            
                            # Set the integration axis
                            integration_axes = tuple([0])

                            # Ensure no other integration axes are specified
                            if not len(rois['z'][z]['roi']['roi_axes']) == 0:
                                raise Exception('Inproper integration axes. You may not specify integration axes on the image.')

                            # Calculate resulting 2d data
                            scale1 = np.average(all_data[f"{rois['z'][z]['req']}_scale1"],axis=0)
                            scale2 = np.average(all_data[f"{rois['z'][z]['req']}_scale2"],axis=0)
                            z_data = stack_roi(all_data[f"{rois['z'][z]['req']}"],xlow,xhigh,idxLow1,idxHigh1,idxLow2,idxHigh2,integration_axes,scale1=scale1,scale2=scale2)
                            
                            # Ensure proper dimensions
                            if len(np.shape(z_data)) == 2:
                                # Add data to locals
                                locals()[f"s{arg}_val{i}_z"] = z_data
                                z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                                # Assign the scales as x/y
                                data[arg].xlabel = f"{rois['z'][z]['req']}_scale1"
                                data[arg].ylabel = f"{rois['z'][z]['req']}_scale2"
                                data[arg].x_data = np.average(all_data[f"{rois['z'][z]['req']}_scale1"],axis=0)[idxLow1:idxHigh1]
                                data[arg].y_data = np.average(all_data[f"{rois['z'][z]['req']}_scale2"],axis=0)[idxLow2:idxHigh2]

                            elif len(np.shape(z_data)) == 1:
                                # This is only to perform additional math operations
                                locals()[f"s{arg}_val{i}_z"] = z_data[:,None]
                                z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                            else:
                                raise Exception(f'Stack ROI ({z}) inproper shape.')

                        else:
                            raise Exception(f'Input dimension wrong ({z})')

                    else:
                        raise Exception(f"Error in specified ROI {rois['z'][z]['roi']} for {z}")
                
                # This is only to perform additional math operations
                elif len(np.shape(all_data[rois['z'][z]['req']])) == 2:
                        # Check that ROI is appropriate
                        if isinstance(rois['z'][z]['roi'],tuple):
                            # Get ROI indices
                            low,high = get_indices(rois['z'][z]['roi'],all_data[f"{rois['z'][z]['req']}_scale"])
                            z_data = mca_roi(all_data[rois['z'][z]['req']],low,high,1,scale=all_data[f"{rois['z'][z]['req']}_scale"])

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_z"] = z_data[:,None]
                            z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                        else:
                            raise Exception(f"Error in specified ROI {rois['z'][z]['roi']} for {z}")

                else:
                    raise Exception(f"Wrong z dimensions ({z})")
                
            # There are no ROI specified
            else:
                # Need to require MCA data with dim = 2
                if len(np.shape(all_data[z])) == 2:
                    # Set the corresponding scale as y-data
                    data[arg].ylabel = f"{z}_scale"
                    data[arg].y_data = all_data[f"{z}_scale"]

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_z"] = all_data[z]
                    z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                elif len(np.shape(all_data[z])) == 1:
                    # This is only to perform additional math operations
                    locals()[f"s{arg}_val{i}_z"] = all_data[z][:,None]
                    z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                else:
                    raise Exception(f"Wrong input dimension {z}")

        # Apply x offset
        data[arg].x_data = apply_offset(data[arg].x_data, xoffset, xcoffset)

        # Apply y offset
        data[arg].y_data = apply_offset(data[arg].y_data, yoffset, ycoffset)

        # Assign the calculated result to the z_stream of the object in data dict
        data[arg].detector = eval(z_stream_convert)

        # Normalize MCA data by SCA
        if not isinstance(norm_by,type(None)):
            norm_data = data[arg].Scan(norm_by)
            normalization = norm_data[norm_by]

            data[arg].detector = detector_norm(data[arg].detector,normalization)

        # Normalize if requested
        if norm == True:
            data[arg].detector = data[arg].detector/np.max(data[arg].detector)

        # Grid data to image
        xmin, xmax, ymin, ymax, new_x, new_y, new_z = grid_data2d(data[arg].x_data, data[arg].y_data, data[arg].detector, grid_x=grid_x,grid_y=grid_y)
        data[arg].xmin = xmin
        data[arg].xmax = xmax
        data[arg].ymin = ymin
        data[arg].ymax = ymax
        data[arg].new_x = new_x
        data[arg].new_y = new_y
        data[arg].new_z = new_z

    return data