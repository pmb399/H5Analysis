"""Processing of scatter data as histogram"""

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
from .simplemath import grid_data_mesh, apply_offset, handle_eval, bin_data, bin_shape_x, bin_shape_1d, grid_data2d

def load_histogram_1d(config, file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize=None, legend_items={}, twin_y=False, matplotlib_props=dict()):
    """ Internal function to generate scatter plots for (x,y) SCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            h5 key or alias of 1d stream
        y_stream: string
            h5 key or alias of 1d, 2d-ROI, or 3d-ROI-ROI stream
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
            binsize: int
                puts x-data in bins of specified size
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
        data[arg].xaxis_label = list()
        data[arg].yaxis_label = list()
        data[arg].filename = file
        data[arg].twin_y = twin_y

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

        # Analyse stream requests with parser
        # Get lists of requisitions
        contrib_x_stream = parse(x_stream)
        contrib_y_stream = parse(y_stream)

        # Strip the requsitions and sort reqs and rois
        reqs, rois = strip_roi(contrib_x_stream,'x', reqs, rois, config)
        reqs, rois = strip_roi(contrib_y_stream,'y', reqs, rois, config)

        # Get the data for all reqs
        all_data = data[arg].Scan(reqs)

        # Check which case (sum or plain) depending on x-data stream
        for i,x in enumerate(contrib_x_stream):
            try:
                # Check if x component has ROI
                if check_key_in_dict(x,rois['x']):
                    if len(np.shape(all_data[rois['x'][x]['req']])) == 1: 
                        # Check that we only have 1 ROI x-stream to reduce to dim 0
                        if len(contrib_x_stream) != 1:
                            raise Exception('Only one ROI x-stream supported.')
                    
                        case_ops = 'sum'

                    else:
                        raise Exception('Data x-stream not supported')

                else:
                    case_ops = 'plain'

            except Exception as e:
                raise Exception(f'Could not get determine histogram type.\nException: {e}')


        if case_ops  == 'plain': # regular histogram (with 1d, 1d inpute)

            x_data = get_hist_stream_1d(contrib_x_stream,x_stream,'x',arg,rois,all_data)
            y_data = get_hist_stream_1d(contrib_y_stream,y_stream,'y',arg,rois,all_data)

            if binsize == None:
                xbins = len(x_data)
            else:
                xbins = int(len(x_data)/binsize)

            hist, bin_edges = np.histogram(x_data,bins=xbins,weights=y_data)

            data[arg].x_stream = np.convolve(bin_edges,[0.5,0.5],'valid') # to get middle of the bins
            data[arg].y_stream = hist
            data[arg].xlabel = f"{x_stream}"
            data[arg].ylabel = f"{y_stream}"
            data[arg].xaxis_label.append(f"{x_stream}")
            data[arg].yaxis_label.append("Intensity")

        elif case_ops == 'sum': # sum over first axis, return scale of 2d array as primary axis
            x_data, roi_tuple = get_hist_stream_0d(contrib_x_stream,x_stream,'x',arg,rois,all_data)
            z_data, scale = get_hist_stream_2d(contrib_y_stream,y_stream,'y',arg,rois,all_data)

            roi_tuple = process_roi_tuple(roi_tuple,x_data)

            SPEC = list()
            for slice1d in np.transpose(z_data): # transpose z_data, so that we go through the data at a specific scale value, for all independent points x_data
                bin,bin_edges = np.histogram(x_data,range=roi_tuple,bins=1,weights=slice1d)
                SPEC.append(bin)

            SPEC = np.array(SPEC)
            # Broadcast  correct shape
            if len(np.shape(SPEC)) == 2 and np.shape(SPEC)[1] == 1:
                SPEC = SPEC[:,0]
            
            data[arg].x_stream = scale
            data[arg].y_stream = SPEC
            data[arg].xlabel = f"{y_stream} Scale"
            data[arg].ylabel = "Intensity"
            data[arg].xaxis_label.append(f"{y_stream} Scale")
            data[arg].yaxis_label.append("Intensity")

            if binsize != None:
                data[arg].x_stream, data[arg].y_stream = bin_data(data[arg].x_stream, data[arg].y_stream, binsize)
        
        else:
            raise Exception('Histogram case undefined.')

        # Apply offsets
        data[arg].x_stream = apply_offset(data[arg].x_stream, xoffset, xcoffset)
        data[arg].y_stream = apply_offset(data[arg].y_stream, yoffset, ycoffset)

        # Normalize if requested
        if norm == True:
            data[arg].z_data = data[arg].z_data / np.max(data[arg].z_data)

        # Do some error checking to ensure matching dimensions
        if len(data[arg].x_stream) != len(data[arg].y_stream):
            raise Exception("Error in x-y stream lengths.")

    return data

def load_histogram_1d_reduce(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize=None, legend_items={}, twin_y=False, matplotlib_props=dict()):
    """ Internal function to generate scatter plots for (x,y,z) SCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            key name or alias of 1d-ROI (dim 0)
        y_stream: string
            key name or alias of 1d-ROI (dim 0)
        z_stream: string
            key name or alias of 2d data stream
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
            binsize: int
                puts x-data in bins of specified size
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
        data[arg].xlabel = f"{z_stream} Scale"
        data[arg].ylabel = "Intensity"
        data[arg].xaxis_label = [f"{z_stream} Scale"]
        data[arg].yaxis_label = ["Intensity"]
        data[arg].filename = file
        data[arg].twin_y = twin_y

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

        x_data, x_roi_tuple = get_hist_stream_0d(contrib_x_stream,x_stream,'x',arg,rois,all_data)
        y_data, y_roi_tuple = get_hist_stream_0d(contrib_y_stream,y_stream,'y',arg,rois,all_data)
        z_data, scale = get_hist_stream_2d(contrib_z_stream,z_stream,'z',arg,rois,all_data)
        
        x_roi_tuple = process_roi_tuple(x_roi_tuple,x_data)
        y_roi_tuple = process_roi_tuple(y_roi_tuple,y_data)

        SPEC = list()
        for slice1d in np.transpose(z_data): # transpose z_data, so that we go through the data at a specific scale value, for all independent points (x_data,y_data)
            bin,xedge,yedge = np.histogram2d(x_data,y_data,range=[x_roi_tuple,y_roi_tuple],bins=1,weights=slice1d)
            SPEC.append(bin)

        SPEC = np.array(SPEC)

        if len(np.shape(SPEC)) == 3 and np.shape(SPEC)[1] == 1 and np.shape(SPEC)[2] == 1:
                SPEC = SPEC[:,0,0]
            
        data[arg].x_stream = scale
        data[arg].y_stream = SPEC

        if binsize != None:
            data[arg].x_stream, data[arg].y_stream = bin_data(data[arg].x_stream, data[arg].y_stream, binsize)
        
        # Apply offsets
        data[arg].x_stream = apply_offset(data[arg].x_stream, xoffset, xcoffset)
        data[arg].y_stream = apply_offset(data[arg].y_stream, yoffset, ycoffset)

        # Normalize if requested
        if norm == True:
            data[arg].z_data = data[arg].z_data / np.max(data[arg].z_data)

        # Do some error checking to ensure matching dimensions
        if len(data[arg].x_stream) != len(data[arg].y_stream):
            raise Exception("Error in x-y stream lengths.")

    return data

def load_histogram_2d(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=None, binsize_y=None, binsize_z=None,bins=None):
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
            binsize_z: int
                puts z-data in bins of specified size
            bins: tuple
                Set the number of bins in the (x-direction,y-direction) explicitly

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

        # Track which case we are analyzing
        # Distinguish three cases
        # (1) (1d,1d,1d)
        # (2) (1d-ROI,1d,2d)
        # (3) (1d-ROI,1d-ROI,3d)

        # Track if there are ROIs in the x and y streams
        # Use variables xroi & yroi

        # Check if there is an ROI in x
        for i,x in enumerate(contrib_x_stream):
            try:
                # Check if x component has ROI
                if check_key_in_dict(x,rois['x']):
                    if len(np.shape(all_data[rois['x'][x]['req']])) == 1: 
                        # Check that we only have 1 ROI x-stream to reduce to dim 0
                        if len(contrib_x_stream) != 1:
                            raise Exception('Only one ROI x-stream supported.')
                        xroi = True
                    else:
                        raise Exception('Data x-stream not supported')
                else:
                    xroi = False
            except Exception as e:
                raise Exception(f'Could not get determine histogram type.\nException: {e}')


        # Check if there is an ROI in y
        for i,y in enumerate(contrib_y_stream):
            try:
                # Check if y component has ROI
                if check_key_in_dict(y,rois['y']):
                    if len(np.shape(all_data[rois['y'][y]['req']])) == 1: 
                        # Check that we only have 1 ROI y-stream to reduce to dim 0
                        if len(contrib_y_stream) != 1:
                            raise Exception('Only one ROI y-stream supported.')
                        yroi = True
                    else:
                        raise Exception('Data y-stream not supported')
                else:
                    yroi = False
            except Exception as e:
                raise Exception(f'Could not get determine histogram type.\nException: {e}')
            
        if xroi == False:
            if yroi == True:
                raise Exception('The ROI needs to be specified in the first argument')
            else:
                # Normal (1d,1d,1d) histogram

                # Get the 1d data
                x_data = get_hist_stream_1d(contrib_x_stream,x_stream_convert,'x',arg,rois,all_data)
                y_data = get_hist_stream_1d(contrib_y_stream,y_stream_convert,'y',arg,rois,all_data)
                z_data = get_hist_stream_1d(contrib_z_stream,z_stream_convert,'z',arg,rois,all_data)

                # Apply offsets
                x_data = apply_offset(x_data, xoffset, xcoffset)
                y_data = apply_offset(y_data, yoffset, ycoffset)

                # Normalize if requested
                if norm == True:
                    z_data = z_data / np.max(z_data)

                # Do some error checking to ensure matching dimensions
                if len(x_data) != len(y_data) or len(y_data) != len(z_data):
                    raise Exception("Error in x-y-z stream lengths.")

                # Calculate the 2d histogram
                new_x, new_y, new_z = grid_data_mesh(x_data,y_data,z_data,binsize_x,binsize_y,bins)

                # Set the labels
                xlabel = x_stream
                ylabel = y_stream
                zlabel = z_stream

        else:
            if yroi == False:
                # Case (0d,1d,2d)

                # Get the data
                x_data,roi_tuple = get_hist_stream_0d(contrib_x_stream,x_stream_convert,'x',arg,rois,all_data)
                y_data = get_hist_stream_1d(contrib_y_stream,y_stream_convert,'y',arg,rois,all_data)
                z_data,scale = get_hist_stream_2d(contrib_z_stream,z_stream_convert,'z',arg,rois,all_data)

                roi_tuple = process_roi_tuple(roi_tuple,x_data)

                # Get the ROI in first coordinate
                roi_idx = (x_data>= roi_tuple[0]) & (x_data<= roi_tuple[1])
                y_data = y_data[roi_idx]
                z_data = z_data[roi_idx,:]

                if binsize_z != None:
                    scale = bin_shape_1d(scale,binsize_z)
                    z_data = bin_shape_x(z_data,binsize_z)

                if binsize_y == None:
                    ybins = len(np.unique(y_data))
                else:
                    ybins = int(len(np.unique(y_data))/binsize_y)

                IMG = list()
                for slice1d in np.transpose(z_data):
                    bin, bin_edges = np.histogram(y_data,bins=ybins,weights=slice1d)
                    IMG.append(bin)

                # Grid data to image
                new_x, new_y, new_z = grid_data2d(np.linspace(bin_edges[0],bin_edges[-1],len(bin_edges)-1), scale, np.array(IMG))

                # Set the labels
                xlabel = y_stream
                ylabel = f"{z_stream} Scale"
                zlabel = z_stream

            else:
                # Case (0d,0d,3d)

                # Get the data
                x_data,roi_tuple1 = get_hist_stream_0d(contrib_x_stream,x_stream_convert,'x',arg,rois,all_data)
                y_data,roi_tuple2 = get_hist_stream_0d(contrib_y_stream,y_stream_convert,'y',arg,rois,all_data)
                z_data,scale1,scale2 = get_hist_stream_3d(contrib_z_stream,z_stream_convert,'z',arg,rois,all_data,config)

                roi_tuple1 = process_roi_tuple(roi_tuple1,x_data)
                roi_tuple2 = process_roi_tuple(roi_tuple2,y_data)

                # Grid data to image
                new_x, new_y, new_z = grid_data2d(np.average(scale1,axis=0), np.average(scale2,axis=0), np.sum(z_data[(x_data>=roi_tuple1[0]) & (x_data<=roi_tuple1[1]) & (y_data>=roi_tuple2[0]) & (y_data<=roi_tuple2[1]),:,:],axis=0))

                # Set the labels
                xlabel = "Scale 1"
                ylabel = "Scale 2"
                zlabel = z_stream

        # Set the values in the object
        data[arg].new_x = new_x
        data[arg].new_y = new_y
        data[arg].new_z = new_z
        data[arg].xlabel = xlabel
        data[arg].ylabel = ylabel
        data[arg].zlabel = zlabel
        data[arg].filename = file

    return data

def load_histogram_2d_sum(config, file, x_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=None, binsize_z=None):
    """ Internal function to generate scatter plots for (x,y) SCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            h5 key or alias of 1d or 1d-ROI stream
        z_stream: string
            h5 key or alias of 2d/3d-ROI or 3d stream
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
            binsize_z: int
                puts z-data in bins of specified size

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
        data[arg].zlabel = z_stream
        data[arg].filename = file

        # Analyse stream requests with parser
        # Get lists of requisitions
        contrib_x_stream = parse(x_stream)
        contrib_z_stream = parse(z_stream)

        # Strip the requsitions and sort reqs and rois
        reqs, rois = strip_roi(contrib_x_stream,'x', reqs, rois, config)
        reqs, rois = strip_roi(contrib_z_stream,'z', reqs, rois, config)

        # Get the data for all reqs
        all_data = data[arg].Scan(reqs)

        # Check which case (sum or plain) depending on x-data stream
        for i,x in enumerate(contrib_x_stream):
            try:
                # Check if x component has ROI
                if check_key_in_dict(x,rois['x']):
                    if len(np.shape(all_data[rois['x'][x]['req']])) == 1: 
                        # Check that we only have 1 ROI x-stream to reduce to dim 0
                        if len(contrib_x_stream) != 1:
                            raise Exception('Only one ROI x-stream supported.')
                    
                        case_ops = '1d-ROI3d'

                    else:
                        raise Exception('Data x-stream not supported')

                else:
                    case_ops = '1d2d'

            except Exception as e:
                raise Exception(f'Could not get determine histogram type.\nException: {e}')


        if case_ops  == '1d2d': # histogram (ind. axis, scale, weights)

            x_data = get_hist_stream_1d(contrib_x_stream,x_stream,'x',arg,rois,all_data)
            z_data, scale = get_hist_stream_2d(contrib_z_stream,z_stream,'z',arg,rois,all_data)

            if binsize_z != None:
                scale = bin_shape_1d(scale,binsize_z)
                z_data = bin_shape_x(z_data,binsize_z)

            if binsize_x == None:
                xbins = len(np.unique(x_data))
            else:
                xbins = int(len(np.unique(x_data))/binsize_x)

            IMG = list()
            for slice1d in np.transpose(z_data):
                bin, bin_edges = np.histogram(x_data,bins=xbins,weights=slice1d)
                IMG.append(bin)

            # Grid data to image
            new_x, new_y, new_z = grid_data2d(np.linspace(bin_edges[0],bin_edges[-1],len(bin_edges)-1), scale, np.array(IMG))

            # Set the label names
            data[arg].xlabel = x_stream
            data[arg].ylabel = f'{z_stream} Scale'

        elif case_ops == '1d-ROI3d':
            x_data, roi_tuple = get_hist_stream_0d(contrib_x_stream,x_stream,'x',arg,rois,all_data)
            z_data, scale1, scale2 = get_hist_stream_3d(contrib_z_stream,z_stream,'z',arg,rois,all_data,config)

            roi_tuple = process_roi_tuple(roi_tuple,x_data)

            # Grid data to image
            new_x, new_y, new_z = grid_data2d(np.average(scale1,axis=0), np.average(scale2,axis=0), np.sum(z_data[(x_data>=roi_tuple[0]) & (x_data<=roi_tuple[1]),:,:],axis=0))

            # Set the label names
            data[arg].xlabel = f'{z_stream} Scale 1'
            data[arg].ylabel = f'{z_stream} Scale 2'

        else:
            raise Exception('Histogram case undefined.')
        
        # Set data
        data[arg].new_x = new_x
        data[arg].new_y = new_y
        data[arg].new_z = new_z

        # Apply offsets
        data[arg].new_x = apply_offset(data[arg].new_x, xoffset, xcoffset)
        data[arg].new_y = apply_offset(data[arg].new_y, yoffset, ycoffset)

        # Normalize if requested
        if norm == True:
            data[arg].new_z = data[arg].new_z / np.max(data[arg].new_z)

    return data


def load_histogram_3d(config, file, x_stream, y_stream, z_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize_x=None, binsize_y=None, binsize_z=None, bins=None):
    """ Internal function to generate scatter plots for (x,y,z) SCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            key name or alias of 1d stream
        y_stream: string
            key name or alias of 1d stream
        z_stream: string
            key name or alias of 2d stream
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
            binsize_z: int
                puts z-data in bins of specified size (along scale direction)
            bins: tuple
                Set the number of bins in the (x-direction,y-direction) explicitly

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
        x_data = get_hist_stream_1d(contrib_x_stream,x_stream_convert,'x',arg,rois,all_data)
        y_data = get_hist_stream_1d(contrib_y_stream,y_stream_convert,'y',arg,rois,all_data)
        z_data, scale = get_hist_stream_2d(contrib_z_stream,z_stream_convert,'z',arg,rois,all_data)

        # Apply offsets
        x_data = apply_offset(x_data, xoffset, xcoffset)
        y_data = apply_offset(y_data, yoffset, ycoffset)

        # Normalize if requested
        if norm == True:
            z_data = z_data / np.max(z_data)

        # Bin along the scale direction of the z-data
        if binsize_z != None:
            z_data = bin_shape_x(z_data,binsize_z)
            scale = bin_shape_1d(scale,binsize_z)

        # Iterate over scale axis and grid 2d histograms
        stack_grid = list()
        new_x_list = list()
        new_y_list = list()

        # Calculate the 2d histogram
        for slice1d in np.transpose(z_data):
            xedge, yedge, new_z = grid_data_mesh(x_data,y_data,slice1d,binsize_x,binsize_y,bins)
            new_x_list.append(xedge)
            new_y_list.append(yedge)
            stack_grid.append(new_z)

        data[arg].new_x = new_x_list
        data[arg].new_y = new_y_list
        data[arg].stack = np.stack(tuple(stack_grid))
        data[arg].ind_stream = scale
        
        data[arg].str_ind_stream = f' {z_stream} Scale'
        data[arg].xlabel = x_stream
        data[arg].ylabel = y_stream
        data[arg].zlabel = z_stream

        data[arg].filename = file

        # Check that independent stream dimension is correct
        if len(data[arg].ind_stream) == np.shape(data[arg].stack)[0]:
            pass
        else:
            raise Exception('Dimension mismatch. Check specified independent stream.')

    return data

def process_roi_tuple(roi_tuple,data):
    """ Processes ROI tuples when None is specified as boundary

        Parameters
        ----------
        roi_tuple: tuple
            Tuple corresponding to region of interest boundaries
        data: array
            1d array data on which the region of interest is specified

        Returns
        -------
        roi_tuple: tuple
            Processed ROI tuple with two floating point numbers
    
    """
        
    if roi_tuple[0] == None:
        entry1 = data[0]
    else:
        entry1 = roi_tuple[0]
    if roi_tuple[1] == None:
        entry2 = data[-1]
    else:
        entry2 = roi_tuple[1]
    return (entry1,entry2)

def get_hist_stream_0d(contrib_stream,convert,stream,arg,rois,all_data):
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
        rois: tuple
            requested (lower,upper) bound
    
    """

    for i,contrib in enumerate(contrib_stream):
        # Check if component has ROI
        if check_key_in_dict(contrib,rois[stream]):

             # Check that dim = 1
            if len(np.shape(all_data[rois[stream][contrib]['req']])) == 1:

                # Check that we only have 1 ROI x-stream to reduce to dim 0
                if len(contrib_stream) != 1:
                    raise Exception('Only one ROI x-stream supported.')
                
                # Add data to locals
                locals()[f"s{arg}_val{i}_{stream}"] = all_data[rois[stream][contrib]['req']]
                convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")

            else:
                raise Exception(f"Wrong {stream} dimensions")

        # No ROI is specified
        else:
            raise Exception(f"Wrong input dimension {contrib}")
                
    return handle_eval(convert,locals()), rois[stream][contrib]['roi']

def get_hist_stream_1d(contrib_stream,convert,stream,arg,rois,all_data):
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

def get_hist_stream_2d(contrib_stream,convert,stream,arg,rois,all_data):
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
        scale: numpy array
            scale of the MCA
    """

    for i,contrib in enumerate(contrib_stream):
        # Check if component has ROI
        if check_key_in_dict(contrib,rois[stream]):
             # Check that dim = 3
            if len(np.shape(all_data[rois[stream][contrib]['req']])) == 3:
                # Check correct ROI type
                if isinstance(rois[stream][contrib]['roi'],dict):
                    # Get indices and reduce data
                    scale1 = np.average(all_data[f"{rois[stream][contrib]['req']}_scale1"],axis=0)
                    scale2 = np.average(all_data[f"{rois[stream][contrib]['req']}_scale2"],axis=0)
                    idxLow1,idxHigh1 = get_indices(rois[stream][contrib]['roi']['roi_list'][0],scale1)
                    idxLow2,idxHigh2 = get_indices(rois[stream][contrib]['roi']['roi_list'][1],scale2)

                    data = stack_roi(all_data[f"{rois[stream][contrib]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,rois[stream][contrib]['roi']['roi_axes'],scale1=scale1,scale2=scale2)

                    # Check correct data dimensions of reduced data
                    if len(np.shape(data)) == 2:
                        # Add data to locals
                        locals()[f"s{arg}_val{i}_{stream}"] = data
                        convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")

                    # Determine the remaining axis
                        integration_axes = rois[stream][contrib]['roi']['roi_axes']
                        all_axes = {1,2}
                        y_axis_raw = all_axes - set(integration_axes)
                        if not len(list(y_axis_raw)) == 1: # Can only have exactly one axis as per dimensionality requirements
                            raise Exception(f'Error determining proper integration axes ({stream})')
                        else:
                            y_axis = list(y_axis_raw)[0] # convert to single element (from set to list, then slice)

                            # Set the remaining (second) axis as y-data
                            if y_axis == 1:
                                scale = np.average(all_data[f"{rois[stream][contrib]['req']}_scale1"],axis=0)[idxLow1:idxHigh1]
                            elif y_axis == 2:
                                scale = np.average(all_data[f"{rois[stream][contrib]['req']}_scale2"],axis=0)[idxLow2:idxHigh2]
                            else:
                                raise Exception(f"Wrong axis defined ({stream}).")
                        
                    else:
                        raise Exception(f'Data dimensionality ({contrib}) incompatible with loader. Check integration axes.')

                else:
                    raise Exception(f"Error in specified ROI {rois[stream][contrib]['roi']['roi_list']} for {contrib}")
            else:
                raise Exception(f"Wrong {stream} dimensions")

        # No ROI is specified
        else:
            # Check correct data dimensions
            if len(np.shape(all_data[contrib])) == 2:
                # Add data to locals
                locals()[f"s{arg}_val{i}_{stream}"] = all_data[contrib]
                convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")
                scale = all_data[f"{contrib}_scale"]

            # This is only to perform additional math operations
            elif len(np.shape(all_data[contrib])) == 1:
                # Add data to locals
                locals()[f"s{arg}_val{i}_{stream}"] = all_data[contrib][:,None]
                convert = convert.replace(contrib,f"s{arg}_val{i}_{stream}")

            else:
                raise Exception(f"Wrong input dimension {contrib}")
                
    return handle_eval(convert,locals()), scale

def get_hist_stream_3d(contrib_stream,convert,stream,arg,rois,all_data,config):
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
        config: dict
        
        Returns
        -------
        data: numpy array
            evaluated
        xscale1: numpy array
            scale of the IMG
        yscale2: numpy array
            scale of the IMG
    """

    has_stack = False

    for i,s in enumerate(contrib_stream):
            # This is only to perform additional math operations
        if check_key_in_dict(s,rois[stream]):
            if len(np.shape(all_data[rois][stream][s]['req'])) == 2:
                low,high = get_indices(rois[stream][s]['roi'],all_data[f"{rois[stream][s]['req']}_scale"])
                data = mca_roi(all_data[rois[stream][s]['req']],low,high,1,scale=all_data[f"{rois[stream][s]['req']}_scale"])

                # Add data to locals
                locals()[f"s{arg}_val{i}_s"] = data[:,None,None]
                convert = convert.replace(s,f"s{arg}_val{i}_s")

            elif len(np.shape(all_data[rois][stream][s]['req'])) == 3:
                if isinstance(rois[stream][s]['roi'],dict):
                    # Get ROI indices
                    scale1 = np.average(all_data[f"{rois[stream][s]['req']}_scale1"],axis=0)
                    scale2 = np.average(all_data[f"{rois[stream][s]['req']}_scale2"],axis=0)
                    idxLow1,idxHigh1 = get_indices(rois[stream][s]['roi']['roi_list'][0],scale1)
                    idxLow2,idxHigh2 = get_indices(rois[stream][s]['roi']['roi_list'][1],scale2)

                    data = stack_roi(all_data[f"{rois[stream][s]['req']}"],None,None,idxLow1,idxHigh1,idxLow2,idxHigh2,(1,2),scale1=scale1,scale2=scale2)

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_s"] = data[:,None,None]
                    convert = convert.replace(s,f"s{arg}_val{i}_s")

                else:
                    raise Exception(f"Error in defined ROI {rois[stream][s]['roi']['roi_list']} for {s}")

            else:
                raise Exception(f"Cannot perform math on defined stream {s}")
            
        else:
            if len(np.shape(all_data[s])) == 3:
                if not check_key_in_dict(s,config.h5dict):
                    raise Exception(f"Data Stream {s} must be configured via config dict.")
                
                if not config.h5dict[s]['type'] == "STACK":
                    raise Exception(f"Need to specify an image stack. Error caused by: {s}")
                
                if has_stack == False:
                    # Get scales
                    xscale1 = all_data[f"{s}_scale1"]
                    yscale2 = all_data[f"{s}_scale2"]

                    # Add data to locals
                    locals()[f"s{arg}_val{i}_s"] = all_data[s]
                    convert = convert.replace(s,f"s{arg}_val{i}_s")

                    has_stack = True

                else:
                    raise Exception('Can only specify one stack')

            elif len(np.shape(all_data[s])) == 1:
                # This is for additional math ops only
                # Add data to locals
                locals()[f"s{arg}_val{i}_s"] = all_data[s][:,None,None]
                convert = convert.replace(s,f"s{arg}_val{i}_s")

            else:
                raise Exception(f'Data dimension of {s} unsupported.')
            
    if has_stack == False:
        raise Exception("No stack specified.")
                   
    return handle_eval(convert,locals()), xscale1, yscale2