from .ReadData import Data
from .datautil import get_roi, get_indices, mca_roi, strip_roi,stack_roi
from .util import check_key_in_dict
from .parser import parse
from .simplemath import apply_offset, grid_data, apply_savgol, bin_data
import warnings

import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp
from numpy import max, min

def load_1d(config, file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None, legend_items={}):

    data = dict()

    for arg in args:
        reqs = list()
        rois = dict()

        data[arg] = Data(config,file,arg)
        data[arg].scan = arg

        contrib_x_stream = parse(x_stream)
        contrib_y_stream = parse(y_stream)

        reqs, rois = strip_roi(contrib_x_stream,'x',reqs, rois)
        reqs, rois = strip_roi(contrib_y_stream,'y',reqs, rois)

        all_data = data[arg].Scan(reqs)

        x_stream_convert = x_stream
        for i,x in enumerate(contrib_x_stream):

            # Check if x component has ROI
            if check_key_in_dict(x,rois['x']):
                # Check that dim(x) = 1
                try:
                    if len(np.shape(all_data[rois['x'][x]['req']])) == 1: 
                        # Check that we only have 1 ROI x-stream
                        if len(contrib_x_stream) != 1:
                            raise Exception('Only one ROI x-stream supported.')
                        if isinstance(rois['x'][x]['roi'],tuple):
                            # Will reduce dim to 0
                            dim_x = 0
                            # Get indices
                            xlow,xhigh = get_indices(rois['x'][x]['roi'],all_data[rois['x'][x]['req']])
                        else:
                            raise Exception("Error in specified ROI")
                    else:
                        raise Exception("Inappropriate dimensions for x-stream")
                except:
                    raise Exception('x-stream undefined.')
            
            # If x component has no ROI
            else:
                try:
                    if len(np.shape(all_data[x])) == 1:
                        # len(contrix_x_stream) requirement above implicitly verifies that
                        # we can only have multiple x components if dim=1
                        dim_x = 1

                        # Add data to locals
                        locals()[f"s{arg}_val{i}_x"] = all_data[x]
                        x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                    else:
                        raise Exception('x-stream dimensions unsupported')
                except:
                    raise Exception('x-stream undefined.')

        if not (dim_x==0 or dim_x == 1):
            raise Exception('Error defining x-stream')
        if dim_x == 1:
            data[arg].x_stream = eval(x_stream_convert)

        y_stream_convert = y_stream
        for i,y in enumerate(contrib_y_stream):
            if check_key_in_dict(y,rois['y']):
                try:
                    # Check that dim(y) = 2
                    if len(np.shape(all_data[rois['y'][y]['req']])) == 2:
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
                            raise Exception("Error in specified ROI")
                    elif len(np.shape(all_data[rois['y'][y]['req']])) == 3:
                        if isinstance(rois['y'][y]['roi'],dict):
                            idxLow1,idxHigh1 = get_indices(rois['y'][y]['roi']['roi_list'][0],all_data[f"{rois['y'][y]['req']}_scale1"])
                            idxLow2,idxHigh2 = get_indices(rois['y'][y]['roi']['roi_list'][1],all_data[f"{rois['y'][y]['req']}_scale2"])

                            if dim_x == 1:
                                y_data = stack_roi(all_data[f"{rois['y'][y]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,rois['y'][y]['roi']['roi_axes'],scale1=all_data[f"{rois['y'][y]['req']}_scale1"],scale2=all_data[f"{rois['y'][y]['req']}_scale2"])
                                if len(np.shape(y_data)) == 1:
                                    # Add data to locals
                                    locals()[f"s{arg}_val{i}_y"] = y_data
                                    y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                                else:
                                    raise Exception('Data dimensionality incompatible with loader. Check integration axes.')

                            elif dim_x == 0:
                                if not isinstance(xlow,type(None)) and not isinstance(xhigh,type(None)):
                                    if xlow > xhigh:
                                        warnings.warn("xlow>xhigh.\nEither select a single value or [None:None] to integrate over all.\nThis most likely happens because the chosen x-stream is not monotonic.")

                                # Add first axis 0 of x-stream to integration
                                integration_axes = tuple([0] + list(rois['y'][y]['roi']['roi_axes']))
                                all_axes = {0,1,2}
                                x_axis_raw = all_axes-set(integration_axes)

                                if not len(list(x_axis_raw)) == 1:
                                    raise Exception('Error determining proper integration axes')
                                else:
                                    x_axis = list(x_axis_raw)[0]
                                    if x_axis == 1:
                                        data[arg].x_stream = all_data[f"{rois['y'][y]['req']}_scale{x_axis}"][idxLow1:idxHigh1]
                                    elif x_axis == 2:
                                        data[arg].x_stream = all_data[f"{rois['y'][y]['req']}_scale{x_axis}"][idxLow2:idxHigh2]
                                    else:
                                        raise Exception("Wrong axis defined.")
                                    y_data = stack_roi(all_data[f"{rois['y'][y]['req']}"],xlow,xhigh,idxLow1,idxHigh1,idxLow2,idxHigh2,integration_axes,scale1=all_data[f"{rois['y'][y]['req']}_scale1"],scale2=all_data[f"{rois['y'][y]['req']}_scale2"])
                                    # Add data to locals
                                    locals()[f"s{arg}_val{i}_y"] = y_data
                                    y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                            else:
                                raise Exception("Incompatible dimensions for chosen x- and y-stream.")

                        else:
                            raise Exception("Error in specified ROI")
                    else:
                        raise Exception("Inappropriate dimensions for y-stream")
                except:
                    raise Exception('y-stream undefined.')
                
            else:
                try:
                    if len(np.shape(all_data[y])) == 1:
                        if dim_x == 1:
                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = all_data[y]
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                        else:
                            raise Exception("x and y have incompatible dimensions")
                    elif len(np.shape(all_data[y])) == 2:
                        if dim_x == 0:
                            data[arg].x_stream = all_data[f"{y}_scale"]
                            y_data = mca_roi(all_data[y],xlow,xhigh,0,scale=all_data[f"{y}_scale"])

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = y_data
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")

                        elif dim_x == 1:
                            y_data = mca_roi(all_data[y],None,None,1,ind_axis=data[arg].x_stream)

                            # Add data to locals
                            locals()[f"s{arg}_val{i}_y"] = y_data
                            y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")

                        else:
                            raise Exception("x and y have incompatible dimensions")
                    else:
                        raise Exception('Improper dimensions of y-stream')

                except:
                    raise Exception('y-stream undefined.')
        
        try:
            data[arg].y_stream = eval(y_stream_convert)
        except:
            raise Exception("Error determining y stream.")
        

        # Get legend items
        try:
            data[arg].legend = legend_items[arg]
        except:
            data[arg].legend = f"S{arg}_{y_stream}"

        #Bin the data if requested
        if binsize != None:
            data[arg].x_stream, data[arg].y_stream = bin_data(data[arg].x_stream,data[arg].y_stream,binsize)

        # Grid the data if specified
        if grid_x != [None, None, None]:
            new_x, new_y = grid_data(
                data[arg].x_stream, data[arg].y_stream, grid_x)

            data[arg].x_stream = new_x
            data[arg].y_stream = new_y

        # Apply offsets to x-stream
        data[arg].x_stream = apply_offset(
        data[arg].x_stream, xoffset, xcoffset)

        # Apply normalization to [0,1]
        if norm == True:
            data[arg].y_stream = np.interp(
                data[arg].y_stream, (data[arg].y_stream.min(), data[arg].y_stream.max()), (0, 1))

        # Apply offset to y-stream
        data[arg].y_stream = apply_offset(
        data[arg].y_stream, yoffset, ycoffset)
               
        # Smooth and take derivatives
        if savgol != None:
            if isinstance(savgol,tuple):
                if len(savgol) == 2: # Need to provide window length and polynomial order
                    savgol_deriv = 0 # Then, no derivative is taken
                elif len(savgol) == 3:
                    savgol_deriv = savgol[2] # May also specify additional argument for derivative order
                else:
                    raise TypeError("Savgol smoothing arguments incorrect.")
                data[arg].x_stream, data[arg].y_stream = apply_savgol(data[arg].x_stream,data[arg].y_stream,savgol[0],savgol[1],savgol_deriv)

                if norm == True:
                    data[arg].y_stream = data[arg].y_stream / \
                    data[arg].y_stream.max()
            else:
                raise TypeError("Savgol smoothing arguments incorrect.")

    return data