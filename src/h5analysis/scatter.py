"""Processing of scatter data as 1d scatter"""

# Data reader
from .ReadData import Data

# Data parser
from .parser import parse

# Utilities
from .datautil import strip_roi

# Simple math OPs
from .histogram import get_hist_stream_1d
from .data_1d import apply_kwargs_1d

def load_scatter_1d(config, file, x_stream, y_stream, *args, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, binsize=None, legend_items={}, twin_y=False, matplotlib_props=dict()):
    """ Internal function to generate scatter plots for (x,y) SCA data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            h5 key or alias of 1d, 2d-ROI, or 3d-ROI-ROI stream
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

        x_data = get_hist_stream_1d(contrib_x_stream,x_stream,'x',arg,rois,all_data)
        y_data = get_hist_stream_1d(contrib_y_stream,y_stream,'y',arg,rois,all_data)

        data[arg].xlabel = f"{x_stream}"
        data[arg].ylabel = f"{y_stream}"
        data[arg].xaxis_label.append(f"{x_stream}")
        data[arg].yaxis_label.append(y_stream)

        data[arg].x_stream, data[arg].y_stream = apply_kwargs_1d(x_data,y_data,norm,xoffset,xcoffset,yoffset,ycoffset,[None, None, None],None,binsize)

        # Do some error checking to ensure matching dimensions
        if len(data[arg].x_stream) != len(data[arg].y_stream):
            raise Exception("Error in x-y stream lengths.")

    return data
