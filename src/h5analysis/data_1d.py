"""Processing of 1d data"""

# Scientific modules and functions
import numpy as np

# Data utility
from .ReadData import Data
from .datautil import get_roi, get_indices, mca_roi, strip_roi,stack_roi

# Utilities
from .util import check_key_in_dict

# Data parser
from .parser import parse

# Simple math OPs
from .simplemath import apply_offset, grid_data, apply_savgol, bin_data, handle_eval

# Warnings
import warnings

def load_1d(config, file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_items={}, twin_y = False, matplotlib_props=dict()):
    """ Internal function to load 1d data

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
        *args: ints
            scan numbers, comma separated
        **kwargs:
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
            grid_x: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            legend_items: dict
                dict[scan number] = description for legend
            twin_y: boolean
                supports a second y-axis on the right-hand side
            matplotlib_props: dict
                dict[scan number] = dict with props, see keys below
                    - linewidth
                    - color
                    - linestyle
                    - marker
                    - markersize
                    - etc.
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
        data[arg].ylabel = y_stream
        data[arg].xaxis_label = list()
        data[arg].yaxis_label = list()
        data[arg].filename = file
        data[arg].twin_y = twin_y

        # Analyse x-stream and y-stream requests with parser
        # Get lists of requisitions
        contrib_x_stream = parse(x_stream)
        contrib_y_stream = parse(y_stream)

        # Strip the requsitions and sort reqs and rois
        reqs, rois = strip_roi(contrib_x_stream,'x',reqs, rois, config)
        reqs, rois = strip_roi(contrib_y_stream,'y',reqs, rois, config)

        # Get the data for all reqs
        all_data = data[arg].Scan(reqs)

        # Set up an x_stream_convert in which we will replace the strings with local data variables
        # and evaluate the expression later
        x_stream_convert = x_stream

        x_idx = False
        if len(contrib_x_stream) == 0:
            # Add special option to plot against index
            dim_x = 1
            x_idx = True

        else:

            # Work through all contributions of x-stream
            for i,x in enumerate(contrib_x_stream):

                # Check if x component has ROI
                if check_key_in_dict(x,rois['x']):
                    # Check that dim(x) = 1
                    try:
                        if len(np.shape(all_data[rois['x'][x]['req']])) == 1: 
                            # Check that we only have 1 ROI x-stream to reduce to dim 0
                            if len(contrib_x_stream) != 1:
                                raise Exception('Only one ROI x-stream supported.')
                            if isinstance(rois['x'][x]['roi'],tuple):
                                # Will reduce dim to 0
                                dim_x = 0
                                # Get indices
                                xlow,xhigh = get_indices(rois['x'][x]['roi'],all_data[rois['x'][x]['req']])
                            else:
                                raise Exception(f"Error in specified ROI {rois['x'][x]['roi']} for {x}")
                        else:
                            raise Exception(f"Inappropriate dimensions for x-stream ({x})")
                    except Exception as e:
                        raise Exception(f'x-stream undefined ({x}).\nException: {e}')
                
                # If x component has no ROI
                else:
                    try:
                        if config.h5dict[x]['x_label'] != None:
                            data[arg].xaxis_label.append(config.h5dict[x]['x_label'])
                    except:
                        pass
                    try:
                        if len(np.shape(all_data[x])) == 1:
                            # len(contrix_x_stream) requirement above implicitly verifies that
                            # we can only have multiple x components if dim=1
                            dim_x = 1

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_x"] = all_data[x]
                            x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                        else:
                            raise Exception(f'x-stream dimensions unsupported ({x})')
                    except Exception as e:
                        raise Exception(f'x-stream undefined ({x}).\nException: {e}')

            # Check proper dimensions for x-stream
            if not (dim_x==0 or dim_x == 1):
                raise Exception('Error defining x-stream')
            
            # If dim_x == 1, can evaluate expression
            if dim_x == 1:
                data[arg].x_stream = handle_eval(x_stream_convert,locals())

        # Set up an y_stream_convert in which we will replace the strings with local data variables
        # and evaluate the expression later
        y_stream_convert = y_stream

        # Work through options for y stream
        for i,y in enumerate(contrib_y_stream):
            # Check if requisition has ROIs
            if check_key_in_dict(y,rois['y']):
                try:
                    if config.h5dict[rois['y'][y]['req']]['y_label'] != None:
                        data[arg].yaxis_label.append(config.h5dict[rois['y'][y]['req']]['y_label'])
                except:
                    pass
                try:
                    # Check that dim(y) = 2
                    if len(np.shape(all_data[rois['y'][y]['req']])) == 2:
                        # Check that ROI is appropriate
                        if isinstance(rois['y'][y]['roi'],tuple):
                            if dim_x == 1:
                                # Get indices and reduce data
                                ylow,yhigh = get_indices(rois['y'][y]['roi'],all_data[f"{rois['y'][y]['req']}_scale"])
                                y_data = mca_roi(all_data[rois['y'][y]['req']],ylow,yhigh,1,scale=all_data[f"{rois['y'][y]['req']}_scale"])
                                # Add data to locals
                                locals()[f"s{arg}_val{i}_y"] = y_data
                                y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                            else:
                                raise Exception('x and y have incompatible dimensions')
                        else:
                            raise Exception(f"Error in specified ROI {rois['y'][y]['roi']} for {y}")
                        
                    # Check that dim(y) = 3
                    elif len(np.shape(all_data[rois['y'][y]['req']])) == 3:
                        # Check that ROI is appropriate
                        if isinstance(rois['y'][y]['roi'],dict):
                            scale1 = np.average(all_data[f"{rois['y'][y]['req']}_scale1"],axis=0)
                            scale2 = np.average(all_data[f"{rois['y'][y]['req']}_scale2"],axis=0)
                            idxLow1,idxHigh1 = get_indices(rois['y'][y]['roi']['roi_list'][0],scale1)
                            idxLow2,idxHigh2 = get_indices(rois['y'][y]['roi']['roi_list'][1],scale2)

                            # Reduce STACK data twice if dim_x == 1
                            if dim_x == 1:
                                y_data = stack_roi(all_data[f"{rois['y'][y]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,rois['y'][y]['roi']['roi_axes'],scale1=scale1,scale2=scale2)
                                # Ensure we reduced to 1d data
                                if len(np.shape(y_data)) == 1:
                                    # Add data to locals
                                    locals()[f"s{arg}_val{i}_y"] = y_data
                                    y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                                else:
                                    raise Exception(f'Data dimensionality of {y} incompatible with loader. Check integration axes.')
                                
                            # Reduce STACK data once if dim_x == 0
                            elif dim_x == 0:
                                if not isinstance(xlow,type(None)) and not isinstance(xhigh,type(None)):
                                    if xlow > xhigh:
                                        warnings.warn("xlow>xhigh.\nEither select a single value or [None:None] to integrate over all.\nThis most likely happens because the chosen x-stream is not monotonic.")

                                # Add first axis 0 of x-stream to integration
                                integration_axes = tuple([0] + list(rois['y'][y]['roi']['roi_axes']))
                                all_axes = {0,1,2}
                                x_axis_raw = all_axes-set(integration_axes) # get new x-stream

                                # Need to have exactly one scale as new x-stream
                                if not len(list(x_axis_raw)) == 1:
                                    raise Exception('Error determining proper integration axes')
                                else:
                                    # Get number of new x-axis
                                    x_axis = list(x_axis_raw)[0] # convert to single element (from set to list, then slice)
                                    if config.h5dict[f"{rois['y'][y]['req']}"][f"label{x_axis}"] != None:
                                        data[arg].xlabel = config.h5dict[f"{rois['y'][y]['req']}"][f"label{x_axis}"]
                                        data[arg].xaxis_label.append(config.h5dict[f"{rois['y'][y]['req']}"][f"label{x_axis}"])
                                    else:
                                        data[arg].xlabel = f"{rois['y'][y]['req']}_scale{x_axis}"
                                        data[arg].xaxis_label.append(f"{rois['y'][y]['req']}_scale{x_axis}")
                                    if x_axis == 1:
                                        data[arg].x_stream = scale1[idxLow1:idxHigh1]
                                    elif x_axis == 2:
                                        data[arg].x_stream = scale2[idxLow2:idxHigh2]
                                    else:
                                        raise Exception("Wrong axis defined.")
                                    
                                    # Reduce data
                                    y_data = stack_roi(all_data[f"{rois['y'][y]['req']}"],xlow,xhigh,idxLow1,idxHigh1,idxLow2,idxHigh2,integration_axes,scale1=scale1,scale2=scale2)
                                    if len(np.shape(y_data)) == 1:
                                        # Add data to locals
                                        locals()[f"s{arg}_val{i}_y"] = y_data
                                        y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                                    else:
                                        raise Exception(f'Data dimensionality of {y} incompatible with loader. Check integration axes.')
                                    
                            else:
                                raise Exception("Incompatible dimensions for chosen x- and y-stream.")

                        else:
                            raise Exception(f"Error in specified ROI {rois['y'][y]['roi']['roi_list']} for {y}")
                    else:
                        raise Exception(f"Inappropriate dimensions for y-stream ({y})")
                except Exception as e:
                    raise Exception(f'y-stream undefined ({y}).\nException: {e}')
                
            # No ROI is specified
            else:
                try:
                    if config.h5dict[y]['y_label'] != None:
                        data[arg].yaxis_label.append(config.h5dict[y]['y_label'])
                except:
                    pass
                try:
                    # stream is 1d
                    if len(np.shape(all_data[y])) == 1:
                        # Ensure we have 1d/1d (x/y) streams
                        if dim_x == 1:
                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = all_data[y]
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                        else:
                            raise Exception("x and y have incompatible dimensions")
                        
                    # Stream is 2d
                    elif len(np.shape(all_data[y])) == 2:
                        # Either dim_x is zero, need to define x-stream with scale
                        if dim_x == 0:
                            if config.h5dict[y]["x_label"] != None:
                                data[arg].xlabel = config.h5dict[y]["x_label"]
                                data[arg].xaxis_label.append(config.h5dict[y]["x_label"])
                            else:
                                data[arg].xlabel = f"{y}_scale"
                                data[arg].xaxis_label.append(f"{y}_scale")
                            data[arg].x_stream = all_data[f"{y}_scale"]
                            # Apply ROI based off x-indices
                            y_data = mca_roi(all_data[y],xlow,xhigh,0,scale=all_data[f"{y}_scale"])

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = y_data
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")

                        # Dimension of x is 1
                        # Reduce the MCA over entire scale range
                        elif dim_x == 1:
                            # Reduce with boundaries None,None for entire range
                            y_data = mca_roi(all_data[y],None,None,1,scale=all_data[f"{y}_scale"])

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = y_data
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")

                        else:
                            raise Exception("x and y have incompatible dimensions")
                        
                    # Stream is 3d - apply automatic reduction
                    elif len(np.shape(all_data[y])) == 3:
                        # Only unambiguous if dim_x == 1 
                        if dim_x == 1:
                            scale1 = np.average(all_data[f"{y}_scale1"],axis=0)
                            scale2 = np.average(all_data[f"{y}_scale2"],axis=0)
                            y_data = stack_roi(all_data[y],None,None,None,None,None,None,(1,2),scale1=scale1,scale2=scale2)

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = y_data
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")

                        else:
                            raise Exception(f'Request ambiguous ({y}).')
                    else:
                        raise Exception(f'Improper dimensions of y-stream {y}')

                except Exception as e:
                    raise Exception(f'y-stream undefined ({y}).\nException" {e}')
        
        try:
            data[arg].y_stream = handle_eval(y_stream_convert,locals())
        except Exception as e:
            raise Exception(f"Error determining y stream.\nException: {e}")
        
        # Calculate the x index based off length of y
        if x_idx == True:
            data[arg].x_stream = np.arange(0,len(data[arg].y_stream))

        # Get legend items
        try:
            data[arg].legend = legend_items[arg]
        except:
            data[arg].legend = f"{config.index}-S{arg}_{x_stream}_{y_stream}"

        # Set matplotlib props
        try:
            data[arg].matplotlib_props = matplotlib_props[arg]
        except:
            data[arg].matplotlib_props = dict()

        data[arg].x_stream,data[arg].y_stream = apply_kwargs_1d(data[arg].x_stream,data[arg].y_stream,norm,xoffset,xcoffset,yoffset,ycoffset,grid_x,savgol,binsize)

    return data

def apply_kwargs_1d(x_stream,y_stream,norm,xoffset,xcoffset,yoffset,ycoffset,grid_x,savgol,binsize):
    """ Internal function to apply math operations as specified in key-word arguments

        Parameters
        ----------
        x_stream: array
            x-data
        y_stream: array
            y-data
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
        grid_x: list
            grid data evenly with [start,stop,delta]
        savgol: tuple
            (window length, polynomial order, derivative)
        binsize: int
            puts data in bins of specified size

        Returns
        -------
        x_stream: array
            Adjusted x-stream
        y_stream: array
            Adjusted y-stream
    """

    #Bin the data if requested
    if binsize != None:
        x_stream, y_stream = bin_data(x_stream,y_stream,binsize)

    # Grid the data if specified
    if grid_x != [None, None, None]:
        new_x, new_y = grid_data(
            x_stream, y_stream, grid_x)

        x_stream = new_x
        y_stream = new_y

    # Apply offsets to x-stream
    x_stream = apply_offset(
    x_stream, xoffset, xcoffset)

    # Apply normalization to [0,1]
    if norm == True:
        y_stream = np.interp(
            y_stream, (y_stream.min(), y_stream.max()), (0, 1))

    # Apply offset to y-stream
    y_stream = apply_offset(
    y_stream, yoffset, ycoffset)
            
    # Smooth and take derivatives
    if savgol != None:
        if isinstance(savgol,tuple):
            if len(savgol) == 2: # Need to provide window length and polynomial order
                savgol_deriv = 0 # Then, no derivative is taken
            elif len(savgol) == 3:
                savgol_deriv = savgol[2] # May also specify additional argument for derivative order
            else:
                raise TypeError("Savgol smoothing arguments incorrect.")
            x_stream, y_stream = apply_savgol(x_stream,y_stream,savgol[0],savgol[1],savgol_deriv)

            if norm == True:
                y_stream = y_stream / y_stream.max()
        else:
            raise TypeError("Savgol smoothing arguments incorrect.")
        
    return x_stream,y_stream