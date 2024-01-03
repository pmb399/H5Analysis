import numpy as np
import warnings
from .ReadData import Data
from .parser import parse
from .datautil import get_roi, get_indices, mca_roi, strip_roi, stack_roi
from .util import check_key_in_dict
from .simplemath import apply_offset, grid_data2d
from .readutil import detector_norm

def load_2d(config, file, x_stream, detector, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None,):
    
    data = dict()
    for arg in args:
        
        reqs = list()
        rois = dict()

        data[arg] = Data(config,file,arg)
        data[arg].scan = arg
        
        contrib_x_stream = parse(x_stream)
        contrib_z_stream = parse(detector)

        reqs, rois = strip_roi(contrib_x_stream,'x', reqs, rois)
        reqs, rois = strip_roi(contrib_z_stream,'z', reqs, rois)

        all_data = data[arg].Scan(reqs)

        # x stream
        x_stream_convert = x_stream
        for i,x in enumerate(contrib_x_stream):
            if check_key_in_dict(x,rois['x']):
                if len(np.shape(all_data[rois['x'][x]['req']])) == 1:
                    if isinstance(rois['x'][x]['roi'],tuple):
                        dim = 0
                        xlow,xhigh = get_indices(rois['x'][x]['roi'],all_data[rois['x'][x]['req']])
                    else:
                        raise Exception('Error in specified ROI')

                elif len(np.shape(all_data[rois['x'][x]['req']])) == 2:
                    if isinstance(rois['x'][x]['roi'],tuple):
                        dim = 1
                        xlow,xhigh = get_indices(rois['x'][x]['roi'],all_data[f"{rois['x'][x]['req']}_scale"])
                        x_data = mca_roi(all_data[rois['x'][x]['req']],xlow,xhigh,1,scale=all_data[f"{rois['x'][x]['req']}_scale"])
                        locals()[f"s{arg}_val{i}_x"] = x_data
                        x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                    else:
                        raise Exception('Error in specified ROI')
                else:
                    raise Exception("Wrong x dimensions")
            else:
                if len(np.shape(all_data[x])) == 1:
                    dim = 1
                    locals()[f"s{arg}_val{i}_x"] = all_data[x]
                    x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                else:
                    raise Exception("Wrong input dimension")
        
        if not (dim==0 or dim == 1):
            raise Exception('Error defining x-stream')
        
        if dim == 1:
            # Assign the calculated result to the x_stream of the object in data dict
            data[arg].x_data = eval(x_stream_convert)
                
        # z-stream
        z_stream_convert = detector
        for i,z in enumerate(contrib_z_stream):
            if check_key_in_dict(z,rois['z']):
                if len(np.shape(all_data[rois['z'][z]['req']])) == 3:
                    if isinstance(rois['z'][z]['roi'],dict):
                        idxLow1,idxHigh1 = get_indices(rois['z'][z]['roi']['roi_list'][0],all_data[f"{rois['z'][z]['req']}_scale1"])
                        idxLow2,idxHigh2 = get_indices(rois['z'][z]['roi']['roi_list'][1],all_data[f"{rois['z'][z]['req']}_scale2"])

                        if dim == 1:
                            z_data = stack_roi(all_data[f"{rois['z'][z]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,rois['z'][z]['roi']['roi_axes'],scale1=all_data[f"{rois['z'][z]['req']}_scale1"],scale2=all_data[f"{rois['z'][z]['req']}_scale2"])
                            if len(np.shape(z_data)) == 2:
                                # Add data to locals
                                locals()[f"s{arg}_val{i}_z"] = z_data
                                z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                                integration_axes = rois['z'][z]['roi']['roi_axes']
                                all_axes = {1,2}
                                y_axis_raw = all_axes - set(integration_axes)
                                if not len(list(y_axis_raw)) == 1:
                                    raise Exception('Error determining proper integration axes')
                                else:
                                    y_axis = list(y_axis_raw)[0]
                                    if y_axis == 1:
                                        data[arg].y_data = all_data[f"{rois['z'][z]['req']}_scale1"][idxLow1:idxHigh1]
                                    elif y_axis == 2:
                                        data[arg].y_data = all_data[f"{rois['z'][z]['req']}_scale2"][idxLow2:idxHigh2]
                                    else:
                                        raise Exception("Wrong axis defined.")
                            else:
                                raise Exception('Data dimensionality incompatible with loader. Check integration axes.')
                            
                        elif dim == 0:
                            if not isinstance(xlow,type(None)) and not isinstance(xhigh,type(None)):
                                if xlow > xhigh:
                                    warnings.warn("xlow>xhigh.\nEither select a single value or [None:None] to integrate over all.\nThis most likely happens because the chosen x-stream is not monotonic.")
                            
                            integration_axes = tuple([0])

                            if not len(rois['z'][z]['roi']['roi_axes']) == 0:
                                raise Exception('Inproper integration axes. You may not specify integration axes on the image.')

                            z_data = stack_roi(all_data[f"{rois['z'][z]['req']}"],xlow,xhigh,idxLow1,idxHigh1,idxLow2,idxHigh2,integration_axes,scale1=all_data[f"{rois['z'][z]['req']}_scale1"],scale2=all_data[f"{rois['z'][z]['req']}_scale2"])
                            if len(np.shape(z_data)) == 2:
                                # Add data to locals
                                locals()[f"s{arg}_val{i}_z"] = z_data
                                z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")

                                data[arg].x_data = all_data[f"{rois['z'][z]['req']}_scale1"][idxLow1:idxHigh1]
                                data[arg].y_data = all_data[f"{rois['z'][z]['req']}_scale2"][idxLow2:idxHigh2]

                            else:
                                raise Exception('Stack ROI inproper shape.')

                        else:
                            raise Exception('Input dimension wrong')

                    else:
                        raise Exception("Error in specified ROI")
                else:
                    raise Exception("Wrong z dimensions")
            else:
                if len(np.shape(all_data[z])) == 2:
                    data[arg].y_data = all_data[f"{z}_scale"]
                    locals()[f"s{arg}_val{i}_z"] = all_data[z]
                    z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")
                else:
                    raise Exception("Wrong input dimension")


        # Apply x offset
        data[arg].x_data = apply_offset(data[arg].x_data, xoffset, xcoffset)

        # Assign the calculated result to the y_stream of the object in data dict
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

        xmin, xmax, ymin, ymax, new_x, new_y, new_z = grid_data2d(data[arg].x_data, data[arg].y_data, data[arg].detector, grid_x=grid_x,grid_y=grid_y)
        data[arg].xmin = xmin
        data[arg].xmax = xmax
        data[arg].ymin = ymin
        data[arg].ymax = ymax
        data[arg].new_x = new_x
        data[arg].new_y = new_y
        data[arg].new_z = new_z

    return data