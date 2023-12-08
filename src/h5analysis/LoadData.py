# Scientific Modules
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d

# Plotting
from bokeh.io import push_notebook
from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, LogColorMapper, ColorBar, Span, Label

# Utilities
import os
from collections import defaultdict
import io
import shutil

# Widgets
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser

# Data Processing Functions
from .util import COLORP
from .add_subtract import ScanAddition, ScanSubtraction, ImageAddition, ImageSubtraction
from .data_1d import loadSCAscans
from .data_2d import loadMCAscans
from .mesh import loadMeshScans
from .beamline_info import loadSCAbeamline, get_single_beamline_value, get_spreadsheet

#########################################################################################
#########################################################################################


class Load1d:
    """Class to load generic 1d (x,y) data."""

    def __init__(self):
        self.data = list()
        self.type = list()
        self.x_stream = list()
        self.filename = list()
        self.plot_lim_x = [":", ":"]
        self.plot_lim_y = [":", ":"]
        self.legend_loc = 'outside'
        self.plot_vlines = list()
        self.plot_hlines = list()
        self.plot_labels = list()

    def load(self, file, x_stream, y_stream, *args, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        file : string
            Specify the file name (HDF5).
        x_stream : string
            Specifiy the data for the horizontal axis.
            Use: "Mono Energy", "MCP Energy", "SDD Energy", "XEOL Energy", "Points", or any SCA scalar array.
        y_stream : string
            Specifiy the data for the vertical axis.
            Use: "TEY", "TFY, "PFY", "iPFY", "XES", "rXES", "specPFY",
                  "XRF", "rXRF", "XEOL", "rXEOL", "POY", "TOY", "EY", "Sample", "Mesh", "ET", or any SCA scalar array.
        *args : int
            Separate scan numbers with comma.
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
                background : int or boolean
                    Apply background selection for XEOL. 
                    Select True when using with getXEOLback function or select background scan number.
                    default : None
                energyloss : boolean or float
                    Convert emission energy to energy loss.
                    Select True to extract incident mono energy from file or specify float manually.
                    default : None
                grid_x: list
                    Grid data evenly on specified grid [low,high,step size]
                    default: [None, None, None]
                savgol : tupel
                    Apply a Savitzky-Golay filter for smoothing, and optionally take derivatives.
                    arg1 : savgol window length
                    arg2 : savgol polynomial order
                    arg 3: derivative order, optional
                    default : None, 
                binsize : int
                    Bin data by reducing data points via averaging.
                    Int must be exponent of 2.
                    default : None
        """

        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", True)
        # Append all REIXS scan objects to scan list in current object.
        self.data.append(loadSCAscans(file, x_stream, y_stream, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.type.append(y_stream)
        self.filename.append(file)

    def add(self, file, x_stream, y_stream, *args, **kwargs):
        """
        Add specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """

        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", False)
        kwargs.setdefault("avg", False)

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(ScanAddition(
            file, x_stream, y_stream, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.type.append(y_stream)
        self.filename.append(file)

    def subtract(self, file, x_stream, y_stream, *args, **kwargs):
        """
        Subtract specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans from the first element. May add scans in first element by specifying list of scans as first *arg.

        """

        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", False)

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(ScanSubtraction(
            file, x_stream, y_stream, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.type.append(y_stream)
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

    def plot(self, linewidth=4, title=None, xlabel=None, ylabel=None, plot_height=450, plot_width=700, y_axis_type='linear'):
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
        """

        # Organize all data assosciated with object in sorted dictionary.
        plot_data = defaultdict(list)
        for i, val in enumerate(self.data):
            for k, v in val.items():
                if len(v.x_stream) != len(v.y_stream):
                    raise UserWarning(f'Error in line {i+1}. Cannot plot (x,y) arrays with differnet size.')
                plot_data["x_stream"].append(v.x_stream)
                plot_data["y_stream"].append(v.y_stream)
                plot_data['x_name'].append(self.x_stream[i])
                plot_data['y_name'].append(self.type[i])
                plot_data['filename'].append(self.filename[i])
                plot_data['scan'].append(v.scan)
                plot_data['legend'].append(v.legend)

        # Get the colours for the glyphs.
        numlines = len(plot_data['scan'])
        plot_data['color'] = COLORP[0:numlines]

        source = ColumnDataSource(plot_data)

        # Set up the bokeh plot
        p = figure(height=plot_height, width=plot_width,y_axis_type=y_axis_type,
                   tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
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
                name = f"~{self.filename[i]}"
                if name not in files:
                    files.append(name)
                fileindex = files.index(name)

                # Append the x_stream data and header name
                series_data.append(pd.Series(v.x_stream))
                series_header.append(f"F{fileindex+1}_S{v.scan}_I{i+1}-{self.x_stream[i]}")

                # Append the y_stream data and header name
                series_data.append(pd.Series(v.y_stream))
                series_header.append(f"F{fileindex+1}_S{v.scan}_I{i+1}-{self.type[i]}")

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

    def load(self, file, x_stream, detector, *args, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        file : string
            Specify the file name (either ASCII or HDF5).
        x_stream : string
            Specifiy the data for the horizontal axis.
            Use: "Mono Energy" or any SCA scalar array.
        y_stream : string
            Specifiy the data for the vertical axis.
            Use: "MCP Energy", "SDD Energy", "XEOL Energy"
        detector : string
            Use: "MCP", "SDD", or "XEOL".
        *args : int
            Specify one (1) scan to load.
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
                background : int or boolean
                    Apply background selection for XEOL. 
                    Select True when using with getXEOLback function or select background scan number.
                    default : None
                energyloss : boolean or float
                    Convert emission energy to energy loss.
                    Select True to extract incident mono energy from file or specify float manually.
                    default : None
                grid_x: list
                    Grid data evenly on specified grid [low,high,step size]
                    default: [None, None, None]
                grid_y: list
                    Grid data evenly on specified grid [low,high,step size]
                    default: [None, None, None]
        """

        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", False)
        kwargs.setdefault("xoffset", None)
        kwargs.setdefault("xcoffset", None)
        kwargs.setdefault("yoffset", None)
        kwargs.setdefault("ycoffset", None)
        kwargs.setdefault("background", None)
        kwargs.setdefault("grid_x", [None, None, None])
        kwargs.setdefault("grid_y", [None, None, None])

        # Ensure that only one scan is loaded.
        if len(args) != 1:
            raise TypeError("You may only select one scan at a time")
        if self.data != []:
            raise TypeError("You can only append one scan per object")
        self.data.append(loadMCAscans(file, x_stream, detector, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.detector.append(detector)
        self.filename.append(file)

        #if kwargs['energyloss'] == True:
        #    self.x_stream[-1] = "Energy loss (eV)"
        #    self.y_stream[-1] = "Mono Energy (eV)"

    def add(self, file, x_stream, detector, *args, **kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """

        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", False)
        kwargs.setdefault("xoffset", None)
        kwargs.setdefault("xcoffset", None)
        kwargs.setdefault("yoffset", None)
        kwargs.setdefault("ycoffset", None)
        kwargs.setdefault("background", None)
        kwargs.setdefault("grid_x", [None, None, None])
        kwargs.setdefault("grid_y", [None, None, None])
        kwargs.setdefault("energyloss", False)

        self.data.append(ImageAddition(file, x_stream,
                         detector, *args, **kwargs))
        
        self.x_stream.append(x_stream)
        self.y_stream.append("Fix it")
        self.detector.append(detector)
        self.filename.append(file)

        if kwargs['energyloss'] == True:
            self.x_stream[-1] = "Energy loss (eV)"
            self.y_stream[-1] = "Mono Energy (eV)"

    def subtract(self, file, x_stream, detector, *args, **kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """
    
        # Set the defaults if not specified in **kwargs.
        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", False)
        kwargs.setdefault("xoffset", None)
        kwargs.setdefault("xcoffset", None)
        kwargs.setdefault("yoffset", None)
        kwargs.setdefault("ycoffset", None)
        kwargs.setdefault("background", None)
        kwargs.setdefault("grid_x", [None, None, None])
        kwargs.setdefault("grid_y", [None, None, None])
        kwargs.setdefault("energyloss", False)

        # Append all REIXS scan objects to scan list in current object.
        self.data.append(ImageSubtraction(file, x_stream,
                         detector, *args, **kwargs))
        
        self.x_stream.append(x_stream)
        self.y_stream.append("Fix it")
        self.detector.append(detector)
        self.filename.append(file)

        if kwargs['energyloss'] == True:
            self.x_stream[-1] = "Energy loss (eV)"
            self.y_stream[-1] = "Mono Energy (eV)"

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
            vmin=None, vmax=None, colormap = "linear"):
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
        """
        # Iterate over the one (1) scan in object - this is for legacy reason and shall be removed in the future.
        for i, val in enumerate(self.data):
            for k, v in val.items():

                # Create the figure
                p = figure(height=plot_height, width=plot_width, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                           tools="pan,wheel_zoom,box_zoom,reset,hover,crosshair,save")
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
                p.yaxis.axis_label = "ToDo" #f"{self.y_stream[i]}"

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

    def load(self, file, x_stream, y_stream, z_stream, *args, **kwargs):

        # Set the defaults if not specified in **kwargs.
        kwargs.setdefault("norm", False)
        kwargs.setdefault("xoffset", None)
        kwargs.setdefault("xcoffset", None)
        kwargs.setdefault("yoffset", None)
        kwargs.setdefault("ycoffset", None)
        #kwargs.setdefault("background", None)
        #kwargs.setdefault("grid_x",[None, None, None])
        #kwargs.setdefault("grid_y",[None, None, None])

        self.data.append(loadMeshScans(file, x_stream,
                         y_stream, z_stream, *args, **kwargs))
        self.x_stream.append(x_stream)
        self.y_stream.append(y_stream)
        self.z_stream.append(z_stream)
        self.filename.append(file)

    def add(self):
        raise UserWarning("Functionality not yet implemented.")
    
    def subtract(self):
        raise UserWarning("Functionality not yet implemented.")
    
    def plot(self, *args, **kwargs):
        kwargs.setdefault('kind', "Histogram")

        super().plot(*args, **kwargs)

    def get_data(self):
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

        f, g, = self.get_data()

        with open(f"{filename}.txt_scale", "a") as scales:
            f.seek(0)
            shutil.copyfileobj(f, scales)

        with open(f"{filename}.txt_matrix", "a") as matrix:
            g.seek(0)
            shutil.copyfileobj(g, matrix)

        print(f"Successfully wrote Histogram data to {filename}.txt")


#########################################################################################


class LoadBeamline(Load1d):
    def load(self, file, key, **kwargs):
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
        self.data.append(loadSCAbeamline(file, key, **kwargs))
        self.type.append(key)
        self.x_stream.append('Scan Number')
        self.filename.append(file)

    def add(*args):
        raise UserWarning('Undefined')

    def subtract(*args):
        raise UserWarning('Undefined')


def getBL(file, stream, *args):
    get_single_beamline_value(file, stream, *args)

def get_spreadsheet(file,columns=None):
    return get_spreadsheet(file,columns)