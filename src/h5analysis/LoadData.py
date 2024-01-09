# Scientific Modules
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d

# Plotting
from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, LogColorMapper, ColorBar, Span, Label
from bokeh.io import push_notebook

# Video Export
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# Utilities
import os
from collections import defaultdict
import io
import shutil
from .util import COLORP

# Widgets
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser

# Data processing functions
from .data_1d import load_1d
from .data_2d import load_2d
from .histogram import load_histogram
from .data_3d import load_3d
from .add_subtract import ScanAddition, ScanSubtraction, ImageAddition, ImageSubtraction, HistogramAddition
from .beamline_info import load_beamline, get_single_beamline_value, get_spreadsheet
from .simplemath import grid_data_mesh

#########################################################################################
#########################################################################################


class Load1d:
    """Class to load generic 1d (x,y) data."""

    def __init__(self):
        self.data = list()
        self.plot_lim_x = [":", ":"]
        self.plot_lim_y = [":", ":"]
        self.legend_loc = 'outside'
        self.plot_vlines = list()
        self.plot_hlines = list()
        self.plot_labels = list()

    def load(self, config, file, x_stream, y_stream, *args, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            h5 key or alias of 1d stream
        y_stream: string
            h5 key or alias of 1d, 2d, or 3d stream
        *args: ints
            scans, comma separated
        **kwargs
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
        """

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(load_1d(config, file, x_stream, y_stream, *args, **kwargs))

    def loadObj(self,obj,line):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        obj: object
            name of the Loader object
        line: int
            Number of the load, add, subtract line (start indexing with 0)
        """

        self.data.append(obj.data[line])

    def background(self,config, file, x_stream, y_stream, arg, **kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            h5 key or alias of 1d stream
        y_stream: string
            h5 key or alias of 1d, 2d, or 3d stream
        arg: int
            scan
        **kwargs
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
        """

        # Get the background data
        background = load_1d(config, file, x_stream, y_stream, arg, **kwargs)
        
        # Subtract the background from all data objects
        for i, val in enumerate(self.data):
            for k, v in val.items():
                # Interpolate the x data onto the background
                new_x = background[arg].x_stream
                int_y = interp1d(v.x_stream,v.y_stream,fill_value='extrapolate')(new_x)

                # Remove data
                new_y = np.subtract(int_y,background[arg].y_stream)

                # Overwrite streams in object
                v.x_stream = new_x
                v.y_stream = new_y

                # Update dictionary with new object
                val[k] = v

            # Update data list with updated dictionary
            self.data[i] = val


    def add(self, config, file, x_stream, y_stream, *args, **kwargs):
        """
        Add specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(ScanAddition(config,
            file, x_stream, y_stream, *args, **kwargs))

    def subtract(self, config, file, x_stream, y_stream, minuend, subtrahend, **kwargs):
        """
        Subtract specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans from the first element. May add scans in first element by specifying list of scans as first *arg.
        """

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(ScanSubtraction(config,
            file, x_stream, y_stream, minuend, subtrahend, **kwargs))

    def xlim(self, lower, upper):
        """
        Set x-axis plot window limits.

        Parameters
        ----------
        lower : float
        upper : float
        """
        self.plot_lim_x[0] = lower
        self.plot_lim_x[1] = upper

    def ylim(self, lower, upper):
        """
        Set y-axis plot window limits.

        Parameters
        ----------
        lower : float
        upper : float
        """
        self.plot_lim_y[0] = lower
        self.plot_lim_y[1] = upper

    def plot_legend(self, pos):
        """
        Overwrite default legend position.

        Parameters
        ----------
        pos : string
            See bokeh manual for available options.
        """
        self.legend_loc = pos

    def vline(self, pos, **kwargs):
        """
        Draw a vertical line in the plot.

        Parameters
        ----------
        pos : float
        **kwargs : dict, optional
            See bokeh manual for available options.
        """
        self.plot_vlines.append([pos, kwargs])

    def hline(self, pos, **kwargs):
        """
        Draw a horizontal line in the plot.

        Parameters
        ----------
        pos : float
        **kwargs : dict, optional
            See bokeh manual for available options.
        """
        self.plot_hlines.append([pos, kwargs])

    def label(self, pos_x, pos_y, text, **kwargs):
        """
        Draw a text box in the plot.

        Parameters
        ----------
        pos_x : float
        pos_y : float
        text : string
        **kwargs : dict, optional
            See bokeh manual for available options.
        """
        self.plot_labels.append([pos_x, pos_y, text, kwargs])

    def plot(self, linewidth=4, title=None, xlabel=None, ylabel=None, plot_height=450, plot_width=700, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        kwargs
            all bokeh figure key-word arguments
        """

        # Organize all data assosciated with object in sorted dictionary.
        plot_data = defaultdict(list)
        for i, val in enumerate(self.data):
            for k, v in val.items():
                if len(v.x_stream) != len(v.y_stream):
                    raise UserWarning(f'Error in line {i+1}. Cannot plot (x,y) arrays with differnet size.')
                plot_data["x_stream"].append(v.x_stream)
                plot_data["y_stream"].append(v.y_stream)
                plot_data['x_name'].append(v.xlabel)
                plot_data['y_name'].append(v.ylabel)
                plot_data['filename'].append(v.filename)
                plot_data['scan'].append(v.scan)
                plot_data['legend'].append(v.legend)

        # Get the colours for the glyphs.
        numlines = len(plot_data['scan'])
        plot_data['color'] = COLORP[0:numlines]

        source = ColumnDataSource(plot_data)

        # Set up the bokeh plot
        p = figure(height=plot_height, width=plot_width,
                   tools="pan,wheel_zoom,box_zoom,reset,crosshair,save", **kwargs)
        p.multi_line(xs='x_stream', ys='y_stream', legend_group="legend",
                     line_width=linewidth, line_color='color', line_alpha=0.6,
                     hover_line_color='color', hover_line_alpha=1.0,
                     source=source)

        # Set up the information for hover box
        p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
            ('Scan', '@scan'),
            ('File', '@filename'),
            ("(x,y)", "(@x_name, @y_name)"),
            ("(x,y)", "($x, $y)")
        ]))

        p.toolbar.logo = None

        # Overwrite plot properties if requested.
        if self.legend_loc == 'outside':
            p.add_layout(p.legend[0], 'right')
        else:
            p.legend.location = self.legend_loc

        if self.plot_lim_y[0] != ':':
            p.y_range.start = self.plot_lim_y[0]
        if self.plot_lim_y[1] != ':':
            p.y_range.end = self.plot_lim_y[1]

        if self.plot_lim_x[0] != ':':
            p.x_range.start = self.plot_lim_x[0]
        if self.plot_lim_x[1] != ':':
            p.x_range.end = self.plot_lim_x[1]

        if len(self.plot_hlines) > 0:
            for line_props in self.plot_hlines:
                line = Span(location=line_props[0],
                            dimension='width', **line_props[1])
                p.add_layout(line)

        if len(self.plot_vlines) > 0:
            for line_props in self.plot_vlines:
                line = Span(
                    location=line_props[0], dimension='height', **line_props[1])
                p.add_layout(line)

        if len(self.plot_labels) > 0:
            for label_props in self.plot_labels:
                label = Label(
                    x=label_props[0], y=label_props[1], text=label_props[2], **label_props[3])
                p.add_layout(label)

        if title != None:
            p.title.text = str(title)
        if xlabel != None:
            p.xaxis.axis_label = str(xlabel)
        if ylabel != None:
            p.yaxis.axis_label = str(ylabel)
        show(p)

    def get_data(self):
        """Make data available in memory as exported to file.

        Returns
        -------
        dfT : pandas DataFrame 
            All loaded data.
        files : list
            List of all loaded files.
        """
        
        files = list()
        series_data = list()
        series_header = list()

        # Iterate over all "load" calls
        for i, val in enumerate(self.data):
            # Iterate over all scans per load call.
            for k, v in val.items():
                name = f"~{v.filename}"
                if name not in files:
                    files.append(name)
                fileindex = files.index(name)

                # Append the x_stream data and header name
                series_data.append(pd.Series(v.x_stream))
                series_header.append(f"F{fileindex+1}_S{v.scan}_I{i+1}-{v.xlabel}")

                # Append the y_stream data and header name
                series_data.append(pd.Series(v.y_stream))
                series_header.append(f"F{fileindex+1}_S{v.scan}_I{i+1}-{v.ylabel}")

        dfT = pd.DataFrame(series_data).transpose(copy=True)
        dfT.columns = series_header

        return dfT, files

    def export(self, filename):
        """
        Export and write data to specified file.

        Parameters
        ----------
        filename : string
        """
        dfT, files = self.get_data()

        # Open file.
        with open(f"{filename}.csv", 'w') as f:
            string = '# '
            # Generate list of files for legend.
            for idx, file in enumerate(files):
                string += f"F{idx+1} {file},"
            string += '\n'
            f.write(string)
            # Write pandas dataframe to file.
            dfT.to_csv(f, index=False, lineterminator='\n')

        print(f"Successfully wrote DataFrame to {filename}.csv")

    def exporter(self):
        """Interactive exporter widget."""
        current_dir = os.path.dirname(os.path.realpath("__file__"))

        self.exportfile = FileChooser(current_dir)
        self.exportfile.use_dir_icons = True
        #self.exportfile.filter_pattern = '*.csv'

        button = widgets.Button(
            description='Save data file',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save data to file',
            icon='save'  # (FontAwesome names without the `fa-` prefix)
        )

        button.on_click(self.exportWidgetStep)
        display(self.exportfile, button)

    def exportWidgetStep(self, my):
        # Helper function for exporter widget.
        file = os.path.join(self.exportfile.selected_path,
                            self.exportfile.selected_filename)
        self.export(file)

#########################################################################################

class Load2d:
    """Class to load generic 2d (x,y,z) image data of a detector."""

    def __init__(self):
        self.data = list()
        self.x_stream = list()
        self.y_stream = list()
        self.detector = list()
        self.filename = list()
        self.plot_lim_x = [":", ":"]
        self.plot_lim_y = [":", ":"]
        self.plot_vlines = list()
        self.plot_hlines = list()
        self.plot_labels = list()

    def load(self, config, file, x_stream, detector, *args, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

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
        arg: int
            scan number
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
        """

        # Ensure that only one scan is loaded.
        if len(args) != 1:
            raise TypeError("You may only select one scan at a time")
        if self.data != []:
            raise TypeError("You can only append one scan per object")
        
        self.data.append(load_2d(config, file, x_stream, detector, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.y_stream.append('Scale')
        self.detector.append(detector)
        self.filename.append(file)

    def background(self,config, file, x_stream, detector, arg, **kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        x_stream: string
            h5 key or alias of 1d stream
        detector: string
            alias of the MCA detector
        arg: int
            scan
        **kwargs
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
        """

        # Get the background data
        background = load_2d(config, file, x_stream, detector, arg, **kwargs)
        
        # Subtract the background from all data objects
        for i, val in enumerate(self.data):
            for k, v in val.items():
                # Interpolate the x,y data onto the background
                new_x = background[arg].new_x
                new_y = background[arg].new_y
                int_z = interp2d(v.new_x,v.new_y,v.new_z)(new_x,new_y)

                # Remove data
                new_z = np.subtract(int_z,background[arg].new_z)

                # Overwrite streams in object
                v.new_x = new_x
                v.new_y = new_y
                v.new_z = new_z

                # Update dictionary with new object
                val[k] = v

            # Update data list with updated dictionary
            self.data[i] = val

    def add(self, config, file, x_stream, detector, *args, **kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """

        # Ensure that only one scan is loaded.
        if len(args) != 1:
            raise TypeError("You may only select one scan at a time")
        if self.data != []:
            raise TypeError("You can only append one scan per object")

        self.data.append(ImageAddition(config,file, x_stream,
                         detector, *args, **kwargs))
        
        self.x_stream.append(x_stream)
        self.y_stream.append("Scale")
        self.detector.append(detector)
        self.filename.append(file)

    def subtract(self, config, file, x_stream, detector, *args, **kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """

        # Ensure that only one scan is loaded.
        if len(args) != 1:
            raise TypeError("You may only select one scan at a time")
        if self.data != []:
            raise TypeError("You can only append one scan per object")

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(ImageSubtraction(config, file, x_stream,
                         detector, *args, **kwargs))
        
        self.x_stream.append(x_stream)
        self.y_stream.append("Scale")
        self.detector.append(detector)
        self.filename.append(file)

    def xlim(self, lower, upper):
        """
        Set x-axis plot window limits.

        Parameters
        ----------
        lower : float
        upper : float
        """
        self.plot_lim_x[0] = lower
        self.plot_lim_x[1] = upper

    def ylim(self, lower, upper):
        """
        Set y-axis plot window limits.

        Parameters
        ----------
        lower : float
        upper : float
        """
        self.plot_lim_y[0] = lower
        self.plot_lim_y[1] = upper

    def vline(self, pos, **kwargs):
        """
        Draw a vertical line in the plot.

        Parameters
        ----------
        pos : float
        **kwargs : dict, optional
            See bokeh manual for available options.
        """
        self.plot_vlines.append([pos, kwargs])

    def hline(self, pos, **kwargs):
        """
        Draw a horizontal line in the plot.

        Parameters
        ----------
        pos : float
        **kwargs : dict, optional
            See bokeh manual for available options.
        """
        self.plot_hlines.append([pos, kwargs])

    def label(self, pos_x, pos_y, text, **kwargs):
        """
        Draw a text box in the plot.

        Parameters
        ----------
        pos_x : float
        pos_y : float
        text : string
        **kwargs : dict, optional
            See bokeh manual for available options.
        """
        self.plot_labels.append([pos_x, pos_y, text, kwargs])

    def plot(self, title=None, kind='Image', xlabel=None, ylabel=None, plot_height=600, plot_width=600, 
            vmin=None, vmax=None, colormap = "linear", **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        kind : string, optional
        xlabel : string, optional
        ylabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        kwargs
            all bokeh figure key-word arguments
        """
        # Iterate over the one (1) scan in object - this is for legacy reason and shall be removed in the future.
        for i, val in enumerate(self.data):
            for k, v in val.items():

                # Create the figure
                p = figure(height=plot_height, width=plot_width, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                           tools="pan,wheel_zoom,box_zoom,reset,hover,crosshair,save", **kwargs)
                p.x_range.range_padding = p.y_range.range_padding = 0

                # Gridded scales now calculated directly during the MCA load and only need to be referenced here

                # must give a vector of image data for image parameter
                if vmin == None:
                    mapper_low = v.new_z.min()
                else:
                    mapper_low = vmin

                if vmax == None:
                    mapper_high = v.new_z.max()
                else:
                    mapper_high = vmax

                if colormap == "linear":
                    myMapper = LinearColorMapper
                elif colormap == "log":
                    myMapper = LogColorMapper
                else:
                    raise UserWarning("Only 'linear' and 'log' implemented.")

                color_mapper = myMapper(palette="Viridis256",
                                                 low=mapper_low,
                                                 high=mapper_high)

                # Plot image and use limits as given by even grid.
                p.image(image=[v.new_z], x=v.xmin, y=v.ymin, dw=v.xmax-v.xmin,
                        dh=v.ymax-v.ymin, color_mapper=color_mapper, level="image")
                p.grid.grid_line_width = 0.5

                # Defining properties of color mapper
                color_bar = ColorBar(color_mapper=color_mapper,
                                     label_standoff=12,
                                     location=(0, 0),
                                     title='Counts')
                p.add_layout(color_bar, 'right')

                # Overwrite plot properties if selected.
                if self.plot_lim_y[0] != ':':
                    p.y_range.start = self.plot_lim_y[0]
                if self.plot_lim_y[1] != ':':
                    p.y_range.end = self.plot_lim_y[1]

                if self.plot_lim_x[0] != ':':
                    p.x_range.start = self.plot_lim_x[0]
                if self.plot_lim_x[1] != ':':
                    p.x_range.end = self.plot_lim_x[1]

                if len(self.plot_hlines) > 0:
                    for line_props in self.plot_hlines:
                        line = Span(
                            location=line_props[0], dimension='width', **line_props[1])
                        p.add_layout(line)

                if len(self.plot_vlines) > 0:
                    for line_props in self.plot_vlines:
                        line = Span(
                            location=line_props[0], dimension='height', **line_props[1])
                        p.add_layout(line)

                if len(self.plot_labels) > 0:
                    for label_props in self.plot_labels:
                        label = Label(
                            x=label_props[0], y=label_props[1], text=label_props[2], **label_props[3])
                        p.add_layout(label)

            if title != None:
                p.title.text = str(title)
            else:
                p.title.text = f'{self.detector[i]} {kind} for Scan {k}'
            if xlabel != None:
                p.xaxis.axis_label = str(xlabel)
            else:
                p.xaxis.axis_label = str(self.x_stream[i])
            if ylabel != None:
                p.yaxis.axis_label = str(ylabel)
            else:
                p.yaxis.axis_label = f'{self.detector[i]} Scale'

            p.toolbar.logo = None

            show(p)

    def get_data(self):
        """Make data available in memory as exported to file.

        Returns
        -------
        f : string.IO object
            Motor and Detector Scales. Pandas Data Series.
            1) Rewind memory with f.seek(0)
            2) Load with pandas.read_csv(f,skiprows=3)
        g : string.IO object
            Actual gridded detector image.
            1) Rewind memory with g.seek(0)
            2) Load with numpy.genfromtxt(g,skip_header=4)
        """
        # Set up the data frame and the two string objects for export
        f = io.StringIO()
        g = io.StringIO()
        series_data = list()
        series_header = list()

        for i, val in enumerate(self.data):
            for k, v in val.items():
                # Gridded scales now calculated directly during the MCA load and only need to be referenced here

                # Start writing string f
                f.write("========================\n")
                f.write(
                    f"F~{self.filename[i]}_S{v.scan}_{self.detector[i]}_{self.x_stream[i]}_{self.y_stream[i]}\n")
                f.write("========================\n")

                # Start writing string g
                g.write("========================\n")
                g.write(
                    f"F~{self.filename[i]}_S{v.scan}_{self.detector[i]}_{self.x_stream[i]}_{self.y_stream[i]}\n")
                g.write("========================\n")

                # Append data to string now.
                # Append x-stream
                series_data.append( pd.Series(v.new_x))
                series_header.append("Motor Scale Gridded")

                # Append y-stream
                series_data.append(pd.Series(v.new_y))
                series_header.append("Detector Scale Gridded")

                dfT = pd.DataFrame(series_data).transpose(copy=True)
                dfT.columns = series_header
                dfT.to_csv(f, index=False, lineterminator='\n')

                g.write("=== Image ===\n")
                np.savetxt(g, v.new_z, fmt="%.9g")

            return f, g

    def export(self, filename):
        """
        Export and write data to specified file.

        Parameters
        ----------
        filename : string
        """
        f, g, = self.get_data()

        # Dump both strings in file.
        # Need to rewind memory location of String.IO to move to beginning.
        # Copy string content to file with shutil.
        with open(f"{filename}.txt_scale", "a") as scales:
            f.seek(0)
            shutil.copyfileobj(f, scales)

        with open(f"{filename}.txt_matrix", "a") as matrix:
            g.seek(0)
            shutil.copyfileobj(g, matrix)

        print(f"Successfully wrote Image data to {filename}.txt")

    def exporter(self):
        """Interactive exporter widget."""
        current_dir = os.path.dirname(os.path.realpath("__file__"))

        self.exportfile = FileChooser(current_dir)
        self.exportfile.use_dir_icons = True
        #self.exportfile.filter_pattern = '*.txt'

        button = widgets.Button(
            description='Save data file',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save data to file',
            icon='save'  # (FontAwesome names without the `fa-` prefix)
        )

        button.on_click(self.exportWidgetStep)
        display(self.exportfile, button)

    def exportWidgetStep(self, my):
        # Helper function for exporter widget
        file = os.path.join(self.exportfile.selected_path,
                            self.exportfile.selected_filename)
        self.export(file)


#########################################################################################

class LoadHistogram(Load2d):
    """Class to display (x,y,z) scatter data."""

    def __init__(self):
        self.data = list()
        self.x_stream = list()
        self.y_stream = list()
        self.z_stream = list()
        self.filename = list()
        self.plot_lim_x = [":", ":"]
        self.plot_lim_y = [":", ":"]
        self.plot_vlines = list()
        self.plot_hlines = list()
        self.plot_labels = list()

        # Use this so we can inherit from Load2d for plotting
        self.detector = self.z_stream

    def load(self, config, file, x_stream, y_stream, z_stream, *args, **kwargs):
        """
        Load (x,y,z) stream data to histogram

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
        args: int
            scan number
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
        """
        
        # Ensure that only one scan is loaded.
        if len(args) != 1:
            raise TypeError("You may only select one scan at a time")
        if self.data != []:
            raise TypeError("You can only append one scan per object")
        
        self.data.append(load_histogram(config, file, x_stream,
                         y_stream, z_stream, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.y_stream.append(y_stream)
        self.z_stream.append(z_stream)
        self.filename.append(file)

    def add(self, config, file, x_stream, y_stream, z_stream, *args, norm=False):
        """
        Add specified histograms for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """

        # Ensure that only one scan is loaded.
        if len(args) != 1:
            raise TypeError("You may only select one scan at a time")
        if self.data != []:
            raise TypeError("You can only append one scan per object")
        
        self.data.append(HistogramAddition(config, file, x_stream,
                         y_stream, z_stream, *args, norm=norm))
        self.x_stream.append(x_stream)
        self.y_stream.append(y_stream)
        self.z_stream.append(z_stream)
        self.filename.append(file)
    
    def subtract(self):
        raise UserWarning("Functionality not yet implemented.")
    
    def plot(self, *args, **kwargs):
        kwargs.setdefault('kind', "Histogram")

        super().plot(*args, **kwargs)

    def get_data(self):
        """Make data available in memory as exported to file.

        Returns
        -------
        f : string.IO object
            Motor and Detector Scales. Pandas Data Series.
            1) Rewind memory with f.seek(0)
            2) Load with pandas.read_csv(f,skiprows=3)
        g : string.IO object
            Actual gridded detector image.
            1) Rewind memory with g.seek(0)
            2) Load with numpy.genfromtxt(g,skip_header=4)
        """

        f = io.StringIO()
        g = io.StringIO()
        for i, val in enumerate(self.data):
            for k, v in val.items():
                # Have the gridded data ready now from loader
                f.write("========================\n")
                f.write(
                    f"F~{self.filename[i]}_S{v.scan}_{self.z_stream[i]}_{self.x_stream[i]}_{self.y_stream[i]}\n")
                f.write("========================\n")

                g.write("========================\n")
                g.write(
                    f"F~{self.filename[i]}_S{v.scan}_{self.z_stream[i]}_{self.x_stream[i]}_{self.y_stream[i]}\n")
                g.write("========================\n")

                f.write("=== x-axis bin edges ===\n")
                np.savetxt(f, v.xedge)
                f.write("=== y-axis bin edges ===\n")
                np.savetxt(f, v.yedge)
                g.write("=== Histogram ===\n")
                np.savetxt(g, v.new_z, fmt="%.9g")
        return f, g

    def export(self, filename):
        """
        Export and write data to specified file.

        Parameters
        ----------
        filename : string
        """
        f, g, = self.get_data()

        with open(f"{filename}.txt_scale", "a") as scales:
            f.seek(0)
            shutil.copyfileobj(f, scales)

        with open(f"{filename}.txt_matrix", "a") as matrix:
            g.seek(0)
            shutil.copyfileobj(g, matrix)

        print(f"Successfully wrote Histogram data to {filename}.txt")


#########################################################################################
        
class Load3d:
    def __init__(self):
        self.data = list()

    def load(self, config, file, stack, arg,**kwargs):
        """ Shows a 3d stack of images interactively

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
        """

        # Ensure we only load 1
        if self.data != []:
            raise UserWarning("Can only load one movie at a time.")
        else:
            self.data.append(load_3d(config, file, stack, arg, **kwargs))

    def plot(self, title=None, xlabel=None, ylabel=None, plot_height=600, plot_width=600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        kwargs
            all bokeh figure key-word arguments
        """

        def update(f=0):
            """Update stack to next image on slider move"""
            r.data_source.data['image'] = [v.stack[f]]
            r.data_source.data['x'] = [v.x_min]
            r.data_source.data['y'] = [v.y_min]
            r.data_source.data['dw'] = [v.x_max-v.x_min]
            r.data_source.data['dh'] = [v.y_max-v.y_min]

            push_notebook(handle=s)

        for i, val in enumerate(self.data):
            for k, v in val.items():
                p = figure(height=plot_height, width=plot_width, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                           tools="pan,wheel_zoom,box_zoom,reset,hover,crosshair,save",**kwargs)
                p.x_range.range_padding = p.y_range.range_padding = 0

                # must give a vector of image data for image parameter
                color_mapper = LinearColorMapper(palette="Viridis256")

                simage = ColumnDataSource(data=dict(image=[v.stack[0]], x=[v.x_min], y=[
                                          v.y_min], dw=[v.x_max-v.x_min], dh=[v.y_max-v.y_min],))

                r = p.image(image='image', source=simage, x='x', y='y',
                            dw='dw', dh='dh', color_mapper=color_mapper, level="image")
                p.grid.grid_line_width = 0.5

                # Defining properties of color mapper
                color_bar = ColorBar(color_mapper=color_mapper,
                                     label_standoff=12,
                                     location=(0, 0),
                                     title='Counts')
                p.add_layout(color_bar, 'right')

                p.toolbar.logo = None

                if title != None:
                    p.title.text = str(title)
                else:
                    p.title.text = f'Image Stack Movie for Scan {k}'
                if xlabel != None:
                    p.xaxis.axis_label = str(xlabel)
                else:
                    pass
                if ylabel != None:
                    p.yaxis.axis_label = str(ylabel)
                else:
                    pass

                s = show(p, notebook_handle=True)
                display(widgets.interact(update, f=(0, len(v.stack)-1)))

    def export(self,filename, interval=500, aspect=1, xlim=None, ylim=None, **kwargs):
        """ Export Stack image as movie

            Parameters
            ----------
            filename: string
            interval: int
                duration of each frame in ms
            aspect: float
                aspect ratio
            xlim: None, tuple
            ylim: None, tuple
            kwargs
                all matplotlib figure key-word arguments
        """

        for i, val in enumerate(self.data):
            for k, v in val.items():
                frames = list()
                fig  = plt.figure(**kwargs)
                if not isinstance(xlim,type(None)):
                    plt.xlim(xlim)
                if not isinstance(ylim,type(None)):
                    plt.ylim(ylim)
                for img in v.stack:
                    frames.append([plt.imshow(img,animated=True,extent=[v.x_min,v.x_max,v.y_min,v.y_max],aspect=aspect)])
            
                ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True,
                                repeat_delay=10000)
                ani.save(filename+'.mp4')

#########################################################################################
class LoadBeamline(Load1d):
    def load(self, config, file, key, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        file : string
            Specify the file name (either ASCII or HDF5).
        key : string
        **kwargs: multiple, optional
            Options:
                norm : boolean
                    Norm the spectra to [0,1].
                    default: True
                xoffset : list of tuples
                    Offset the x-axis by applying a polynomial fit.
                    default: None
                xcoffset : float
                    Offset x-axis by constant value.
                    default : None 
                yoffset : list of tuples
                    Offset the y-axis by applying a polynomial fit.
                    default : None 
                ycoffset : float
                    Offset y-axis by constant value.
                    default : None
        """

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(load_beamline(config, file, key, **kwargs))

    def add(*args):
        raise UserWarning('Undefined')

    def subtract(*args):
        raise UserWarning('Undefined')


def getBL(config, file, stream, *args):
    """Load beamline meta data.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        keys: string, list
            path to the meta data of interest
        args: int
            scan numbers, comma separated
    """
    get_single_beamline_value(config, file, stream, *args)

def getSpreadsheet(config, file,columns=None):
    """Generate spreadsheet with meta data from h5 file.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        columns: dict
            Specify column header and h5 data path to meta datam i.e.
                columns = dict()
                columns['Sample Stage horz'] = 'Endstation/Motors/ssh
                ...
    """
    return get_spreadsheet(config, file,columns)

#########################################################################################
#########################################################################################