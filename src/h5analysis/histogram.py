from .ReadData import Data
from .parser import parse
from .datautil import get_roi, get_indices, mca_roi, strip_roi
from .util import check_key_in_dict
from .simplemath import grid_data_mesh, apply_offset

import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp
from numpy import max, min


def load_histogram(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None):
    """Internal function to generate scatter plots for (x,y,z) SCA data"""

    # Generate dictionary to store data
    data = dict()
    for arg in args:
        reqs = list()
        rois = dict()

        # Load scans to dict
        data[arg] = Data(config,file,arg)
        data[arg].scan = arg

        contrib_x_stream = parse(x_stream)
        contrib_y_stream = parse(y_stream)
        contrib_z_stream = parse(z_stream)

        reqs, rois = strip_roi(contrib_x_stream,'x', reqs, rois)
        reqs, rois = strip_roi(contrib_y_stream,'y', reqs, rois)
        reqs, rois = strip_roi(contrib_z_stream,'z', reqs, rois)

        all_data = data[arg].Scan(reqs)

        # x stream
        x_stream_convert = x_stream
        for i,x in enumerate(contrib_x_stream):
            if check_key_in_dict(x,rois['x']):
                if len(np.shape(all_data[rois['x'][x]['req']])) == 2:
                    if isinstance(rois['x'][x]['roi'],tuple):
                        xlow,xhigh = get_indices(rois['x'][x]['roi'],all_data[f"{rois['x'][x]['req']}_scale"])
                        x_data = mca_roi(all_data[rois['x'][x]['req']],xlow,xhigh,1,scale=all_data[f"{rois['x'][x]['req']}_scale"])
                        locals()[f"s{arg}_val{i}_x"] = x_data
                        x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                    else:
                        raise Exception('Error in specified ROI')
                elif len(np.shape(all_data[rois['x'][x]['req']])) == 3:
                    if isinstance(rois['x'][x]['roi'],list):
                        raise Exception("Not implemented")
                    else:
                        raise Exception("Error in specified ROI")
                else:
                    raise Exception("Wrong x dimensions")
            else:
                if len(np.shape(all_data[x])) == 1:
                    locals()[f"s{arg}_val{i}_x"] = all_data[x]
                    x_stream_convert = x_stream_convert.replace(x,f"s{arg}_val{i}_x")
                else:
                    raise Exception("Wrong input dimension")
                
        # y stream
        y_stream_convert = y_stream
        for i,y in enumerate(contrib_y_stream):
            if check_key_in_dict(y,rois['y']):
                if len(np.shape(all_data[rois['y'][y]['req']])) == 2:
                    if isinstance(rois['y'][y]['roi'],tuple):
                        ylow,yhigh = get_indices(rois['y'][y]['roi'],all_data[f"{rois['y'][y]['req']}_scale"])
                        y_data = mca_roi(all_data[rois['y'][y]['req']],ylow,yhigh,1,scale=all_data[f"{rois['y'][y]['req']}_scale"])
                        locals()[f"s{arg}_val{i}_y"] = y_data
                        y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                    else:
                        raise Exception("Error in specified ROI")
                elif len(np.shape(all_data[rois['y'][y]['req']])) == 3:
                    if isinstance(rois['y'][y]['roi'],list):
                        raise Exception("Not implemented")
                    else:
                        raise Exception("Error in specified ROI")
                else:
                    raise Exception("Wrong y dimensions")
            else:
                if len(np.shape(all_data[y])) == 1:
                    locals()[f"s{arg}_val{i}_y"] = all_data[y]
                    y_stream_convert = y_stream_convert.replace(y,f"s{arg}_val{i}_y")
                else:
                    raise Exception("Wrong input dimension")
                
        # z stream
        z_stream_convert = z_stream
        for i,z in enumerate(contrib_z_stream):
            if check_key_in_dict(z,rois['z']):
                if len(np.shape(all_data[rois['z'][z]['req']])) == 2:
                    if isinstance(rois['z'][z]['roi'],tuple):
                        zlow,zhigh = get_indices(rois['z'][z]['roi'],all_data[f"{rois['z'][z]['req']}_scale"])
                        z_data = mca_roi(all_data[rois['z'][z]['req']],zlow,zhigh,1,scale=all_data[f"{rois['z'][z]['req']}_scale"])
                        locals()[f"s{arg}_val{i}_z"] = z_data
                        z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")
                    else:
                        raise Exception("Error in specified ROI")
                elif len(np.shape(all_data[rois['z'][z]['req']])) == 3:
                    if isinstance(rois['z'][z]['roi'],list):
                        raise Exception("Not implemented")
                    else:
                        raise Exception("Error in specified ROI")
                else:
                    raise Exception("Wrong z dimensions")
            else:
                if len(np.shape(all_data[z])) == 1:
                    locals()[f"s{arg}_val{i}_z"] = all_data[z]
                    z_stream_convert = z_stream_convert.replace(z,f"s{arg}_val{i}_z")
                else:
                    raise Exception("Wrong input dimension")


        # Assign the calculated result to the x_stream of the object in data dict
        data[arg].x_data = eval(x_stream_convert)
        data[arg].x_data = apply_offset(data[arg].x_data, xoffset, xcoffset)

        # Assign the calculated result to the y_stream of the object in data dict
        data[arg].y_data = eval(y_stream_convert)
        data[arg].y_data = apply_offset(data[arg].y_data, yoffset, ycoffset)

        # Assign the calculated result to the z_stream of the object in data dict
        data[arg].z_data = eval(z_stream_convert)

        # Normalize if requested
        if norm == True:
            data[arg].z_data = np.interp(
                data[arg].z_data, (data[arg].z_data.min(), data[arg].z_data.max()), (0, 1))

        xmin, xmax, ymin, ymax, xedge, yedge, new_z, zmin, zmax = grid_data_mesh(data[arg].x_data,data[arg].y_data,data[arg].z_data)
        data[arg].xmin = xmin
        data[arg].xmax = xmax
        data[arg].ymin = ymin
        data[arg].ymax = ymax
        data[arg].xedge = xedge
        data[arg].yedge = yedge
        data[arg].new_z = new_z
        data[arg].zmin = zmin
        data[arg].zmax = zmax

    return data
