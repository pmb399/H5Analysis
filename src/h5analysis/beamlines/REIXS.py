from h5analysis.LoadData import *
from h5analysis.MathData import *
from h5analysis.config import h5Config
from bokeh.io import show, output_notebook
output_notebook(hide_banner=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

## Add h5 extension, remove all others
def hdf5FileFixer(hdf5_filename):
    if hdf5_filename.find('.h5') < 0:
        file_ext = hdf5_filename.split('.')
        hdf5_filename = file_ext[0] + '.h5'
    return hdf5_filename

## REIXS specific log viewer class to truncate or refine the log data
class LogLoader():
    """Generate spreadsheet with meta data from h5 file."""
    def load(self,config,filename,logbook,ncols=[]):
        """Load meta data from h5 file.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            filename
        logbook: dict
            Specify column configuration for specific REIXS endstation
        **kwargs: optional
            Options:
                ncols: string list
                    determines which columns are ecluded in the logbook from columns dictionary
        """
        ## Fix filename if needed
        filename = hdf5FileFixer(filename)
        if len(ncols):
           cols_mod = dict()
           for col in logbook:
                if col in ncols:
                    pass
                else:
                    cols_mod[col] = logbook[col]
        else:
           cols_mod = logbook
        self.df = get_spreadsheet(config,filename,cols_mod,average=True)
    ## Change the text in a cell
    def replace(self, column, scan, text):
        """Replace logbook entry

        Parameters
        ----------
        column: string
            Column title of the cell to which replace
        scan: int
            Scan number of cell to which replace
        text: string
            Text to replace current text in selected cell
        """
        self.df.loc[scan,column] = text
    ## Sort by field value, either == (implied), !, <, >
    def filter(self, column, value):
        """Replace logbook entry

        Parameters
        ----------
        column: string
            Column title of the cell to which replace
        value: string
            Entry value to replace with condition - !, >, or less (== implied)
        """
        col_val = value.split('!')
        if(len(col_val) > 1):
            ## Try to convert to float
            try:
                sort_val = float(col_val[1])
            except:
                sort_val = col_val[1]
            self.df = self.df.loc[self.df[column] != sort_val]
            return
        col_val = value.split('>')
        if(len(col_val) > 1):
            ## Try to convert to float
            try:
                sort_val = float(col_val[1])
                self.df = self.df.loc[self.df[column] > sort_val]
                return
            except:
                return
        col_val = value.split('<')
        if(len(col_val) > 1):
            ## Try to convert to float
            try:
                sort_val = float(col_val[1])
                self.df = self.df.loc[self.df[column] < sort_val]
                return
            except:
                return
        ## Using default ==
        try:
            sort_val = float(value)
        except:
            sort_val = value
        self.df = self.df.loc[self.df[column] == sort_val]
    ## Add addition text to cell
    def append(self, column, scan, text):
        """Append to logbook entry

        Parameters
        ----------
        column: string
            Column title of the cell for which to append
        scan: int
            Scan number of cell for which to append
        text: string
            Text to append in selected cell
        """
        self.df.loc[scan,column] = self.df[column][scan] + ' ' + text
    #Allow the table to reduced to a specific scan range
    def show(self, scans=[]):
        """Display the spreadsheet as pandas DataFrame
        **kwargs: optional
            Options:
                scans: int list - one or two entries
                    determins the lower and upper bound of scan entries, upper not required
                    NOTE: The export option will override scans option
        
        """
        if len(scans) == 2:
            #Remove enertries from spreadsheet
            self.df = self.df.truncate(before = scans[0], after = scans[1])
            return self.df
        elif len(scans) == 1:
            self.df = self.df.truncate(before = scans[0])
            return self.df
        else:
            return self.df
    def export(self,filename):
        """Export the created spreadsheet in csv format as log file.

        Parameters
        ----------
        filename: string
            file name of the created csv file
        """

        self.df.to_csv(f"{filename}.csv")
        return self.df
        
#Make plots of metadata as a function of scan
class BeamlineLoader(LoadBeamline):
    """Load meta data as 1d data stream."""
    def load(self, config, filename, metadata, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            filename
        metadata : string
        **kwargs: multiple, optional
            Options:
                norm : boolean
                    Norm the spectra to [0,1].
                    default: True
                yoffset : list of tuples
                    Offset the y-axis by applying a polynomial fit.
                    default : None 
                ycoffset : float
                    Offset y-axis by constant value.
                    default : None
                twin_y: boolean
                    supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        LoadBeamline.load(self, config, filename, metadata, **kwargs)
    def plot(self, linewidth = 2, xlabel ='Scan #', ylabel='Value',ylabel_right='Value',plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        LoadBeamline.plot(self, linewidth= linewidth, xlabel =xlabel, ylabel=ylabel,ylabel_right=ylabel_right, plot_width = plot_width, plot_height = plot_height,**kwargs)

class SimpleLoader(Load1d):
    """Class to load generic 1d data as a function of scan #."""
    def load(self,config,filename,y_stream,*args,**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        y_stream: string
            h5 key or alias of 1d, 2d, or 3d stream
        *args: ints
            scans, comma separated
        **kwargs
            norm: boolean
                normalizes to [0,1]
            yoffset: list
                fitting offset (y-stream)
            ycoffset: float
                constant offset (y-stream)
            twin_y: boolean
                supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        Load1d.load(self,config,filename,'',y_stream,*args,**kwargs)
    def add(self,config,filename,y_stream,*args,**kwargs):
        """
        Add specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load1d.add(self,config,filename,'',y_stream,*args,**kwargs)
    def subtract(self,config,filename,y_stream,*args,**kwargs):
        """
        Subtract specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans from the first element. May add scans in first element by specifying list of scans as first *arg.
        """
        filename = hdf5FileFixer(filename)
        Load1d.subtract(self,config,filename,'',y_stream,*args,**kwargs)
    def stitch(self,config,filename,y_stream,*args,**kwargs):
        """
        Stitch specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        return Load1d.stitch(self,config,filename,'',y_stream,*args,**kwargs)
    def background(self,config,filename,y_stream,*args,**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        y_stream: string
            h5 key or alias of 1d, 2d, or 3d stream
        *args: int
            scans
        **kwargs
            norm: boolean
                normalizes to [0,1]
            yoffset: list
                fitting offset (y-stream)
            ycoffset: float
                constant offset (y-stream)
        """
        filename = hdf5FileFixer(filename)
        Load1d.background(self,config,filename,'',y_stream,*args,**kwargs)
    def compare(self,plot_object):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        plot_object: object
            name of the Loader object
        """
        for i in range(len(plot_object.data)):
            Load1d.loadObj(self, plot_object, i)
    def plot(self, linewidth = 2,xlabel ='Point #', ylabel='Value',ylabel_right='Value', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        Load1d.plot(self, linewidth = linewidth, xlabel = xlabel, ylabel = ylabel,ylabel_right=ylabel_right, plot_width = plot_width, plot_height = plot_height,**kwargs)

class PlotLoader(Load1d):
    """Class to load generic 1d (x,y) data."""
    def load(self,config,filename,x_stream, y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            twin_y: boolean
                supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        Load1d.load(self,config,filename,x_stream,y_stream,*args,grid_x=grid,**kwargs)
    def fit(self, line, scans, center, amplitude, sigma, fit = 'best', report = False,lim = [None,None], **kwargs):
        for j in range(len(scans)):
            Fit_Spectra = Object1dFit()
            Fit_Spectra.load(self,line,scans[j])    
            for i in range(len(center)):
                print(center[i])
                Fit_Spectra.add_Gaussian(center[i],amplitude[i],sigma[i],**kwargs)
            Fit_Spectra.evaluate(fit=fit, lower_limit=lim[0], upper_limit=lim[1])
            if report==True:
                Fit_Spectra.fit_report()
                Fit_Spectra.fit_values()
            Load1d.loadObj(self, Fit_Spectra, 1)
    def add(self,config,filename,x_stream, y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Add specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load1d.add(self,config,filename,x_stream,y_stream,*args,grid_x=grid,**kwargs)
    def subtract(self,config,filename,x_stream, y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Subtract specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans from the first element. May add scans in first element by specifying list of scans as first *arg.
        """
        filename = hdf5FileFixer(filename)
        Load1d.subtract(self,config,filename,x_stream,y_stream,*args,grid_x=grid,**kwargs)
    def background(self,config,filename,x_stream, y_stream,*args,grid=[None, None, None],**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        x_stream: string
            h5 key or alias of 1d stream
        y_stream: string
            h5 key or alias of 1d, 2d, or 3d stream
        *args: int
            scans
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
        """
        filename = hdf5FileFixer(filename)
        Load1d.background(self,config,filename,x_stream,y_stream,*args,grid_x=grid,**kwargs)
    def stitch(self,config,filename,x_stream, y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Stitch specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load1d.stitch(self,config,filename,x_stream,y_stream,*args,grid_x=grid,**kwargs)
    def compare(self,plot_object):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        plot_object: object
            name of the Loader object
        """
        for i in range(len(plot_object.data)):
            Load1d.loadObj(self, plot_object, i)
    def plot(self, linewidth = 2, xlabel = '', ylabel = 'Value',ylabel_right = 'Value', plot_width = 900, plot_height = 600,**kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        total_xlabel = ''
        for i in range(len(self.data)):
            for j in self.data[i].keys():
                if self.data[i][j].xlabel not in total_xlabel:
                    if len(total_xlabel) > 0:
                         total_xlabel += '/ '
                    total_xlabel += self.data[i][j].xlabel + ' '
        if len(xlabel) == 0:
            xlabel = total_xlabel
        Load1d.plot(self, linewidth = linewidth, xlabel = xlabel, ylabel = ylabel, ylabel_right=ylabel_right,plot_width = plot_width, plot_height = plot_height, **kwargs)

class ImageLoader(Load2d):
    """Class to load generic 2d (x,y,z) image data of a detector."""
    def load(self,config,filename,x_stream, detector,*args, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            filename
        x_stream: string
            h5 sca key or alias of the x-stream
        detector: string
            alias of the MCA detector
        arg: int
            scan number
        **kwargs
            norm: boolean
                Can be boolean or None (as False)
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize_y: int
                puts data in bins of specified size in the vertical direction
        """
        filename = hdf5FileFixer(filename)
        Load2d.load(self,config,filename,x_stream,detector, *args, **kwargs)
    def add(self,config,filename,x_stream, detector,*args, **kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load2d.add(self,config,filename,x_stream,detector,*args, **kwargs)
    def subtract(self,config,filename,x_stream, detector,*args,**kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """
        filename = hdf5FileFixer(filename)
        Load2d.subtract(self,config,filename,x_stream,detector,*args,**kwargs)
    def background_1d(self,config,filename,x_stream, detector,*args,grid=[None, None, None],**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        x_stream: string
            h5 key or alias of 1-d stream
        detector: string
            h5 key or alias of 1-d reduced stream
        *args: int
            scans
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
        """
        filename = hdf5FileFixer(filename)
        Load2d.background_1d(self,config,filename,x_stream,detector,*args,grid_x=grid,**kwargs)
    def background_2d(self,config,filename,x_stream, detector,*args,**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        x_stream: string
            h5 key or alias of 1d stream
        detector: string
            alias of the MCA detector
        *args: int
            scans
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
                grid equally spaced in x with [start, stop, delta]
            grid_y: list
                grid equally spaced in y with [start, stop, delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            legend_items: dict
                dict[scan number] = description for legend
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize_y: int
                puts data in bins of specified size in the vertical direction
        """
        filename = hdf5FileFixer(filename)
        Load2d.background_2d(self,config,filename,x_stream,detector,*args,**kwargs)
    def stitch(self,config,filename,x_stream, detector,*args,**kwargs):
        """
        Stitch specified scans for selected image.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load2d.stitch(self,config,filename,x_stream,detector,*args,**kwargs)
    def plot(self, plot_width = 900, plot_height = 600,**kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        zlabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        norm : boolean
            to normalize the plot to the maximum
        kwargs
            all bokeh figure key-word arguments
        """
        Load2d.plot(self, plot_width = plot_width, plot_height = plot_height, **kwargs) 
    
class StackLoader(Load3d):
    """Object to hold a 3d stack of images"""
    def load(self,config,filename,scan_stream, detector,*args, **kwargs):
        """ Shows a 3d stack of images interactively

            Parameters
            ----------
            config: dict
                REIXS beamline endstation configuration
            filename: string
                filename
            scan_stream: string
                independent scanned stream, corresponding to stack's first dim
            detector: string
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
        filename = hdf5FileFixer(filename)
        Load3d.load(self,config,filename,scan_stream,detector, *args, **kwargs)
    def add(self,config,filename,scan_stream, detector,*args, **kwargs):
        """ Adds 3d stacks of images with identical scales

            Parameters
            ----------
            See Load3d function.
            Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load3d.add(self,config,filename,scan_stream,detector,*args, **kwargs)
    def subtract(self,config,filename,scan_stream, detector,*args,**kwargs):
        """ Subtracts 3d stacks of images with identical scales

            Parameters
            ----------
            See Load3d function, but
            minuend: list
                adds all images in list, generates minuend
            subtrahend: list
                adds all images in list, generates subtrahend
        """
        filename = hdf5FileFixer(filename)
        Load3d.subtract(self,config,filename,scan_stream,detector,*args,**kwargs)
    def plot(self, plot_width = 900, plot_height = 600,**kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        zlabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        norm: boolean
            Normalizes to the maximum z-value across all images in the stack
        kwargs
            all bokeh figure key-word arguments
        """
        Load3d.plot(self, plot_width = plot_width, plot_height = plot_height, **kwargs) 
    
class MeshLoader(LoadHistogram):
    """Class to display (x,y,z) scatter data."""
    def load(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs):
        """
        Load (x,y,z) stream data to histogram

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize_y: int
                puts data in bins of specified size in the vertical direction
        """
        filename = hdf5FileFixer(filename)
        LoadHistogram.load(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs)
    def add(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs):
        """
        Add specified histograms for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        LoadHistogram.add(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs)
    def subtract(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs):
        """
        Subract specified histograms for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans specified in two *args lists.
        """
        filename = hdf5FileFixer(filename)
        LoadHistogram.subtract(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs)
    def stitch(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs):        
        """
        Stitch specified scans for selected histograms.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        LoadHistogram.stitch(self, config, filename, x_stream, y_stream, z_stream, *args, **kwargs)
    def plot(self, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        zlabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        norm : boolean
            to normalize the plot to the maximum
        kwargs
            all bokeh figure key-word arguments
        """
        LoadHistogram.plot(self, plot_width = 900, plot_height = 600,**kwargs)

###################################    
## ENERGY SPECIFIC REIXS LOADERS ##
###################################

class XESLoader(Load1d):
    """ Load and plot X-ray emission scan(s) """
    def load(self,config,filename,detector,*args,grid=[None, None, None],**kwargs):
        """
        Load one or multiple scan(s).

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        detector: string
            alias for MCA type detector
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            twin_y: boolean
                supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            detector = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        Load1d.load(self,config,filename,x_stream,detector,*args,grid_x=grid,**kwargs)
    ## Fit data with gaussians
    def fit(self, line, scans, center, amplitude, sigma, fit = 'best', report = False,lim = [None,None], center_bound=[],sigma_bound=[],amplitude_bound=[],**kwargs):
        for j in range(len(scans)):
            Fit_Spectra = Object1dFit()
            Fit_Spectra.load(self,line,scans[j])    
            for i in range(len(center)):
                if(len(center_bound) == len(center)):
                    center_bounds = center_bound[i]
                else:
                    center_bounds = (None,None)
                if(len(sigma_bound) == len(center)):
                    sigma_bounds = sigma_bound[i]
                else:
                    sigma_bounds = (None,None)
                if(len(amplitude_bound) == len(center)):
                    amplitude_bounds = amplitude_bound[i]
                else:
                    amplitude_bounds = (None,None)
                Fit_Spectra.add_Gaussian(center[i],amplitude[i],sigma[i],center_bounds = center_bounds,sigma_bounds=sigma_bounds,amplitude_bounds=amplitude_bounds,**kwargs)
            Fit_Spectra.evaluate(fit=fit, lower_limit=lim[0], upper_limit=lim[1])
            if report==True:
                Fit_Spectra.fit_report()
                Fit_Spectra.fit_values()
            XESLoader.loadObj(self, Fit_Spectra, 1)
    def stitch(self,config,filename,detector,*args,grid=[None, None, None],**kwargs):
        """
        Stitch specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = xes_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            xes_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        Load1d.stitch(self,config,filename,x_stream,detector,*args,grid_x=grid,**kwargs)
    def add(self,config,filename,detector,*args,grid=[None, None, None],**kwargs):
        """
        Add specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            detector = data_args[0]+data_args2[1]
            
        else:
            x_stream = '[None]'
        Load1d.add(self,config,filename,x_stream,detector,*args,grid_x=grid,**kwargs)
    def subtract(self,config,filename,detector,*args,grid=[None, None, None],**kwargs):
        """
        Subtract specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans from the first element. May add scans in first element by specifying list of scans as first *arg.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            detector = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        Load1d.subtract(self,config,filename,x_stream,detector,*args,grid_x=grid,**kwargs)
    def background(self,config,filename,detector,*args,grid=[None, None, None],**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        detector: string
            alias for MCA type detector
        *args: int
            scans
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            legend_items: dict
                dict[scan number] = description for legend
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            detector = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        Load1d.background(self,config,filename,x_stream,detector,*args,grid_x=grid,**kwargs)
    def compare(self,plot_object):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        plot_object: object
            name of the Loader object
        """
        for i in range(len(plot_object.data)):
            XESLoader.loadObj(self, plot_object, i)
    def plot(self, linewidth = 2, xlabel='Emission Energy [eV]', ylabel='Counts',  ylabel_right='Counts', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        Load1d.plot(self, linewidth = linewidth, xlabel=xlabel, ylabel=ylabel,ylabel_right=ylabel_right,plot_width = plot_width, plot_height = plot_height,**kwargs)
        
class XEOLLoader(XESLoader):
    """ Load and plot X-ray excited optical luminescence scan(s) """
    def plot(self, linewidth = 2, xlabel='Wavelength [nm]', ylabel='Counts', ylabel_right = 'Counts', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        Load1d.plot(self, linewidth = linewidth, xlabel=xlabel, ylabel=ylabel,ylabel_right=ylabel_right, plot_width = plot_width, plot_height = plot_height,**kwargs)
   
    
class XASLoader(Load1d):
    """ Load and plot X-ray absorption scan(s) """
    def load(self,config,filename,y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file namem
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            twin_y: boolean
                supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        Load1d.load(self,config,filename,'Energy',y_stream,*args,grid_x=grid,**kwargs)
    def add(self,config,filename,y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Add specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load1d.add(self,config,filename,'Energy',y_stream,*args,grid_x=grid,**kwargs)
    def subtract(self,config,filename,y_stream,*args,grid=[None, None, None],**kwargs):
        """
        Subtract specified scans for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all scans from the first element. May add scans in first element by specifying list of scans as first *arg.
        """
        filename = hdf5FileFixer(filename)
        Load1d.subtract(self,config,filename,'Energy',y_stream,*args,grid_x=grid,**kwargs)
    def background(self,config,filename,y_stream,*args,grid=[None, None, None],**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
           REIXS beamline endstation configuration
        filename: string
            file name
        y_stream: string
            h5 key or alias of 1d, 2d, or 3d stream
        *args: int
            scans
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
        """
        filename = hdf5FileFixer(filename)
        Load1d.background(self,config,filename,'Energy',y_stream,*args,grid_x=grid,**kwargs)
    def compare(self,plot_object):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        plot_object: object
            name of the Loader object
        """
        for i in range(len(plot_object.data)):
            XASLoader.loadObj(self, plot_object, i)
    def plot(self, linewidth = 2, xlabel='Excitation Energy [eV]', ylabel='Relative Intensity',ylabel_right= 'Relative Intensity', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        Load1d.plot(self, linewidth = linewidth, xlabel=xlabel, ylabel=ylabel,ylabel_right=ylabel_right, plot_width = plot_width, plot_height = plot_height,**kwargs)

class XESMapper(Load2d):
    """ Load and plot X-ray emission map """
    def load(self,config,filename,detector, args,**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            filename
        detector: string
            alias of the MCA detector
        arg: int
            scan number
        **kwargs
            norm: boolean
                Can be boolean or None (as False)
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize_y: int
                puts data in bins of specified size in the vertical direction
        """
        filename = hdf5FileFixer(filename)
        Load2d.load(self,config,filename,'Energy',detector, args,**kwargs)
    def add(self,config,filename,detector,*args,**kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load2d.add(self,config,filename,'Energy',detector,*args,**kwargs)
    def stitch(self,config,filename,detector,*args,**kwargs):
        """
        Stitch specified scans for selected image.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        Load2d.stitch(self,config,filename,'Energy',detector,*args,**kwargs)
    def subtract(self,config,filename,detector,*args,**kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """
        filename = hdf5FileFixer(filename)
        Load2d.subtract(self,config,filename,'Energy',detector,*args,**kwargs)
    def background_1d(self,config,filename,detector,*args,grid= [None,None,None],**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        detector: string
            h5 key or alias of 1d stream
        *args: int
            scans
        **kwargs
            axis: string
                <<x>> or <<y>> axis for subtraction direction
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
        """
        filename = hdf5FileFixer(filename)
        Load2d.background_1d(self,config,filename,'[None]',detector,*args,grid_x=grid,**kwargs)
    def background_2d(self,config,filename,detector,*args,**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            filename
        detector: string
            alias of the MCA detector
        *args: int
            scans
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
            grid_y: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
                dict[scan number] = description for legend
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize_y: int
                puts data in bins of specified size in the vertical direction
        """
        filename = hdf5FileFixer(filename)
        Load2d.background_2d(self,config,filename,'Energy',detector,*args,**kwargs)
    def plot(self, xlabel='Excitation Energy [eV]', ylabel='Emission Energy [eV]', plot_width = 900, plot_height = 600,**kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        kind : string, optional
        xlabel : string, optional
        ylabel : string, optional
        zlabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        norm : boolean
            to normalize the plot to the maximum
        kwargs
            all bokeh figure key-word arguments
        """
        Load2d.plot(self, xlabel=xlabel, ylabel=ylabel, plot_width = plot_width, plot_height = plot_height,**kwargs)
 
class XEOLMapper(XESMapper):
    """ Load and plot X-ray excited optical luminescence map """
    def plot(self, xlabel='Excitation Energy [eV]', ylabel='Wavelength [nm]', plot_width = 900, plot_height = 600,**kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        kind : string, optional
        xlabel : string, optional
        ylabel : string, optional
        zlabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        norm : boolean
            to normalize the plot to the maximum
        kwargs
            all bokeh figure key-word arguments
        """
        Load2d.plot(self, xlabel=xlabel, ylabel=ylabel, plot_width = plot_width, plot_height = plot_height,**kwargs)
    
class PFYLoader(Object2dReduce):
    """ Load and plot complex x-ray absorption scan(s) """
    def load(self,config,filename, detector,*args, grid=[None,None,None],binsize = None, **kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        detector: string
            alias of the MCA detector
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            twin_y: boolean
                supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        for scan_i in args:
            #Load Image for processing
            PFY_Image = Load2d()
            PFY_Image.load(config,filename,'Energy',detector,scan_i)
            #Load total 1D spectra for intensity comparison
            PFY_Spectra = Load1d()
            PFY_Spectra.load(config,filename,'Energy',detector,scan_i)
            #Make another image, but use it for normalization
            XAS_Image = Object2dReduce()
            XAS_Image.load(PFY_Image,0,scan_i)
            #Reference is the max of the 1D data
            max_old = PFY_Spectra.data[0][scan_i].y_stream.max()
            #Mx of  the interopolated image and compare for 1D max
            XAS_Image.roi('y', roi = (None,None))
            max_new = XAS_Image.data[0][0].y_stream.max()
            PFY_Image = Load2d()
            PFY_Image.load(config,filename,'Energy',detector,scan_i, binsize_x = binsize,grid_x=grid,**kwargs)
            Object2dReduce.load(self, PFY_Image,0,scan_i)
            if(len(data_args) > 1):
                data_args2 = data_args1[0].split(',')
                if(len(data_args2) > 1):
                    y_range1 = data_args2[0].split(':')
                    y_min1 = float(min(y_range1))
                    y_max1 = float(max(y_range1))
                    x_min = float(min(PFY_Image.data[0][scan_i].new_x))
                    y_range2 = data_args2[1].split(':')
                    y_min2 = float(min(y_range2))
                    y_max2 = float(max(y_range2))
                    x_max = float(max(PFY_Image.data[0][scan_i].new_x))
                    Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
                else:
                    y_range = data_args2[0].split(':')
                    y_min = float(min(y_range))
                    y_max = float(max(y_range))
                    Object2dReduce.roi(self,'y', roi =  (y_min,y_max))
            else:
                Object2dReduce.roi(self,'y', roi =  (None,None))
            scale_fact = max_old/max_new
            self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
            self.data[len(self.data)-1][0].legend = str(len(self.data)) + '-S' + str(scan_i) + '_Energy_' + str(y_stream_orig)
    def stitch(self,config,filename, detector,*args, binsize = None,grid=[None,None,None], **kwargs):
        """
        Stitch specified scans for selected image.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        #Load Image for processing
        PFY_Image = Load2d()
        PFY_Image.stitch(config,filename,'Energy',detector, *args)
        #Load total 1D spectra for intensity comparison
        PFY_Spectra = Load1d()
        PFY_Spectra.stitch(config,filename,'Energy',detector, *args)
        #Make another image, but use it for normalization
        XAS_Image = Object2dReduce()
        XAS_Image.load(PFY_Image,0,0)
        #Reference is the max of the 1D data
        max_old = max(PFY_Spectra.data[0][0].y_stream)
        #Mx of  the interopolated image and compare for 1D max
        XAS_Image.roi('y', roi = (None,None))
        max_new = XAS_Image.data[0][0].y_stream.max()
        PFY_Image = Load2d()
        PFY_Image.stitch(config,filename,'Energy',detector, *args, binsize_x = binsize,grid_x=grid,**kwargs)
        Object2dReduce.load(self, PFY_Image,0,0)
        if(len(data_args) > 1):
            data_args2 = data_args1[0].split(',')
            if(len(data_args2) > 1):
                y_range1 = data_args2[0].split(':')
                y_min1 = float(min(y_range1))
                y_max1 = float(max(y_range1))
                x_min = float(min(PFY_Image.data[0][0].new_x))
                y_range2 = data_args2[1].split(':')
                y_min2 = float(min(y_range2))
                y_max2 = float(max(y_range2))
                x_max = float(max(PFY_Image.data[0][0].new_x))
                Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
            else:
                y_range = data_args2[0].split(':')
                y_min = float(min(y_range))
                y_max = float(max(y_range))
                Object2dReduce.roi(self,'y', roi =  (y_min,y_max))
        else:
            Object2dReduce.roi(self,'y', roi =  (None,None))
        scale_fact = max_old/max_new
        self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
        self.data[len(self.data)-1][0].legend = str(len(self.data))+'-S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '+' + str(args[i])
        self.data[len(self.data)-1][0].legend += '_Energy_' + str(y_stream_orig)
    def add(self,config,filename, detector,*args, binsize = None, grid=[None,None,None],**kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        #Load Image for processing
        PFY_Image = Load2d()
        PFY_Image.add(config,filename,'Energy',detector, *args)
        #Load total 1D spectra for intensity comparison
        PFY_Spectra = Load1d()
        PFY_Spectra.add(config,filename,'Energy',detector, *args)
        #Make another image, but use it for normalization
        XAS_Image = Object2dReduce()
        XAS_Image.load(PFY_Image,0,0)
        #Reference is the max of the 1D data
        max_old = PFY_Spectra.data[0][0].y_stream.max()
        #Mx of  the interopolated image and compare for 1D max
        XAS_Image.roi('y', roi = (None,None))
        max_new = XAS_Image.data[0][0].y_stream.max()
        PFY_Image = Load2d()
        PFY_Image.add(config,filename,'Energy',detector, *args, binsize_x = binsize,grid_x=grid,**kwargs)
        Object2dReduce.load(self, PFY_Image,0,0)
        if(len(data_args) > 1):
            data_args2 = data_args1[0].split(',')
            if(len(data_args2) > 1):
                y_range1 = data_args2[0].split(':')
                y_min1 = float(min(y_range1))
                y_max1 = float(max(y_range1))
                x_min = float(min(PFY_Image.data[0][0].new_x))
                y_range2 = data_args2[1].split(':')
                y_min2 = float(min(y_range2))
                y_max2 = float(max(y_range2))
                x_max = float(max(PFY_Image.data[0][0].new_x))
                Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
            else:
                y_range = data_args2[0].split(':')
                y_min = float(min(y_range))
                y_max = float(max(y_range))
                Object2dReduce.roi(self,'y', roi =  (y_min,y_max))
        else:
            Object2dReduce.roi(self,'y', roi =  (None,None))
        scale_fact = max_old/max_new
        self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
        self.data[len(self.data)-1][0].legend = str(len(self.data))+'-S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '+' + str(args[i])
        self.data[len(self.data)-1][0].legend += '_Energy_' + str(y_stream_orig)
    def subtract(self,config,filename, y_stream,*args, binsize = None, grid=[None,None,None],**kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """
        filename = hdf5FileFixer(filename)
        data_args = y_stream.split('[')
        y_stream_orig = y_stream
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            y_stream = data_args[0] + data_args1[1]
        else:
            y_stream = data_args[0]
        #Load Image for processing
        PFY_Image = Load2d()
        PFY_Image.subtract(config,filename,'Energy',y_stream, *args)
        #Load total 1D spectra for intensity comparison
        PFY_Spectra = Load1d()
        PFY_Spectra.subtract(config,filename,'Energy',y_stream, *args)
        #Make another image, but use it for normalization
        XAS_Image = Object2dReduce()
        XAS_Image.load(PFY_Image,0,0)
        #Reference is the max of the 1D data
        max_old = PFY_Spectra.data[0][0].y_stream.max()
        #Mx of  the interopolated image and compare for 1D max
        XAS_Image.roi('y', roi = (None,None))
        max_new = XAS_Image.data[0][0].y_stream.max()
        PFY_Image = Load2d()
        PFY_Image.subtract(config,filename,'Energy',y_stream, *args, binsize_x = binsize,grid_x=grid,**kwargs)
        Object2dReduce.load(self, PFY_Image,0,0)
        if(len(data_args) > 1):
            data_args2 = data_args1[0].split(',')
            if(len(data_args2) > 1):
                y_range1 = data_args2[0].split(':')
                y_min1 = float(min(y_range1))
                y_max1 = float(max(y_range1))
                x_min = float(min(PFY_Image.data[0][0].new_x))
                y_range2 = data_args2[1].split(':')
                y_min2 = float(min(y_range2))
                y_max2 = float(max(y_range2))
                x_max = float(max(PFY_Image.data[0][0].new_x))
                Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
            else:
                y_range = data_args2[0].split(':')
                y_min = float(min(y_range))
                y_max = float(max(y_range))
                Object2dReduce.roi(self,'y', roi =  (y_min,y_max))
        else:
            Object2dReduce.roi(self,'y', roi =  (None,None))
        scale_fact = max_old/max_new
        self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
        self.data[len(self.data)-1][0].legend = str(len(self.data))+'-S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '-' + str(args[i])
        self.data[len(self.data)-1][0].legend += '_Energy_' + str(y_stream_orig)
    def compare(self,plot_object):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        plot_object: object
            name of the Loader object
        """
        for i in range(len(plot_object.data)):
            Load1d.loadObj(self, plot_object, i)
    def plot(self, linewidth = 2, xlabel='Excitation Energy [eV]', ylabel='Relative Intensity', ylabel_right='Relative Intensity', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        Load1d.plot(self, linewidth = linewidth, xlabel=xlabel, ylabel=ylabel,ylabel_right=ylabel_right, plot_width = plot_width, plot_height = plot_height,**kwargs)


class ELOSSLoader(Object2dReduce):
    """ Load and plot complex x-ray emission scan(s) on energy loss scale """
    def load(self,config,filename, detector,*args, binsize = None,grid=[None,None,None],**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            file name
        detector: string
            alias of the MCA detector
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
            grid: list
                grid data evenly with [start,stop,delta]
            savgol: tuple
                (window length, polynomial order, derivative)
            binsize: int
                puts data in bins of specified size
            twin_y: boolean
                supports a second y-axis on the right-hand side
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        for scan_i in args:
            #Load Image for processing
            RIXS_Image = Object2dTransform()
            RIXS_Image.load(config,filename,'Energy',detector,scan_i)
            #Load total 1D spectra for intensity comparison
            RIXS_Spectra = Load1d()
            RIXS_Spectra.load(config,filename,'[None]',detector,scan_i)
            #Make another image, but use it for normalization
            XES_Image = Object2dReduce()
            XES_Image.load(RIXS_Image,0,scan_i)
            #Reference is the max of the 1D data
            max_old = RIXS_Spectra.data[0][scan_i].y_stream.max()
            #Mx of  the interopolated image and compare for 1D max
            XES_Image.roi('x', roi = (None,None))
            max_new = XES_Image.data[0][0].y_stream.max()
            RIXS_Image = Object2dTransform()
            RIXS_Image.load(config,filename,'Energy',detector,scan_i, binsize_y = binsize,grid_x=grid,**kwargs)
            if(len(data_args) > 1):
                data_args2 = data_args1[0].split(':')
                x_min = float(min(data_args2))
                x_max = float(max(data_args2))
                RIXS_Image.transform('x-y', xlim = (x_min,x_max))
                Object2dReduce.load(self, RIXS_Image,0, scan_i)
                Object2dReduce.roi(self,'x', roi =  (x_min,x_max))
            else:
                RIXS_Image.transform('x-y')
                Object2dReduce.load(self, RIXS_Image,0, scan_i)
                Object2dReduce.roi(self,'x', roi =  (None,None))
            scale_fact = max_old/max_new

            self.data[len(self.data)-1][scan_i] = self.data[len(self.data)-1][0]
            del self.data[len(self.data)-1][0]
            self.data[len(self.data)-1][scan_i].y_stream = self.data[len(self.data)-1][scan_i].y_stream*scale_fact
            self.data[len(self.data)-1][scan_i].legend = str(len(self.data)) + '-S' + str(scan_i) + '_Energy_' + str(y_stream_orig)
    def fit(self, line, scan, center, amplitude, sigma, fit = 'best', report = False,lim = [None,None], center_bound=[],sigma_bound=[],amplitude_bound=[],**kwargs):
        for j in range(len(line)):
            Fit_Spectra = Object1dFit()
            Fit_Spectra.load(self,line[j],scan)    
            for i in range(len(center)):
                if(len(center_bound) == len(center)):
                    center_bounds = center_bound[i]
                else:
                    center_bounds = (None,None)
                if(len(sigma_bound) == len(center)):
                    sigma_bounds = sigma_bound[i]
                else:
                    sigma_bounds = (None,None)
                if(len(amplitude_bound) == len(center)):
                    amplitude_bounds = amplitude_bound[i]
                else:
                    amplitude_bounds = (None,None)
                Fit_Spectra.add_Gaussian(center[i],amplitude[i],sigma[i],center_bounds = center_bounds,sigma_bounds=sigma_bounds,amplitude_bounds=amplitude_bounds,**kwargs)
            Fit_Spectra.evaluate(fit=fit, lower_limit=lim[0], upper_limit=lim[1])
            if report==True:
                Fit_Spectra.fit_report()
            Load1d.loadObj(self, Fit_Spectra, 1)
    def stitch(self,config,filename, detector,*args, binsize = None,grid=[None,None,None], **kwargs):
        """
        Stitch specified scans for selected image.

        Parameters
        ----------
        See loader function.
        stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        #Load Image for processing
        RIXS_Image = Object2dTransform()
        RIXS_Image.stitch(config,filename,'Energy',detector,*args)
        #Load total 1D spectra for intensity comparison
        RIXS_Spectra = Load1d()
        RIXS_Spectra.stitch(config,filename,'[None]',detector, *args)
        #Make another image, but use it for normalization
        XES_Image = Object2dReduce()
        XES_Image.load(RIXS_Image,0,0)
        #Reference is the max of the 1D data
        max_old = max(RIXS_Spectra.data[0][0].y_stream)
        #Mx of  the interopolated image and compare for 1D max
        XES_Image.roi('x', roi = (None,None))
        max_new = XES_Image.data[0][0].y_stream.max()
        RIXS_Image = Object2dTransform()
        RIXS_Image.stitch(config,filename,'Energy',detector,*args, binsize_y = binsize,grid_x=grid,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args1[0].split(':')
            x_min = float(min(data_args2))
            x_max = float(max(data_args2))
            RIXS_Image.transform('x-y', xlim = (x_min,x_max))
            Object2dReduce.load(self, RIXS_Image,0, 0)
            Object2dReduce.roi(self,'x', roi =  (x_min,x_max))
        else:
            RIXS_Image.transform('x-y')
            Object2dReduce.load(self, RIXS_Image,0, 0)
            Object2dReduce.roi(self,'x', roi =  (None,None))
        scale_fact = max_old/max_new
        self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
        self.data[len(self.data)-1][0].legend = str(len(self.data))+'-S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '+' + str(args[i])
        self.data[len(self.data)-1][0].legend += '_Energy_' + str(y_stream_orig)
        return Object2dReduce
    def add(self,config,filename, detector,*args, binsize = None,grid=[None,None,None], **kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        #Load Image for processing
        RIXS_Image = Object2dTransform()
        RIXS_Image.add(config,filename,'Energy',detector,*args)
        #Load total 1D spectra for intensity comparison
        RIXS_Spectra = Load1d()
        RIXS_Spectra.add(config,filename,'[None]',detector, *args)
        #Make another image, but use it for normalization
        XES_Image = Object2dReduce()
        XES_Image.load(RIXS_Image,0,0)
        #Reference is the max of the 1D data
        max_old = RIXS_Spectra.data[0][0].y_stream.max()
        #Mx of  the interopolated image and compare for 1D max
        XES_Image.roi('x', roi = (None,None))
        max_new = XES_Image.data[0][0].y_stream.max()
        RIXS_Image = Object2dTransform()
        RIXS_Image.add(config,filename,'Energy',detector,*args,binsize_y = binsize,grid_x=grid, **kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args1[0].split(':')
            x_min = float(min(data_args2))
            x_max = float(max(data_args2))
            RIXS_Image.transform('x-y', xlim = (x_min,x_max))
            Object2dReduce.load(self, RIXS_Image,0, 0)
            Object2dReduce.roi(self,'x', roi =  (x_min,x_max))
        else:
            RIXS_Image.transform('x-y')
            Object2dReduce.load(self, RIXS_Image,0, 0)
            Object2dReduce.roi(self,'x', roi =  (None,None))
        scale_fact = max_old/max_new
        self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
        self.data[len(self.data)-1][0].legend = str(len(self.data))+'-S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '+' + str(args[i])
        self.data[len(self.data)-1][0].legend += '_Energy_' + str(y_stream_orig)
        return Object2dReduce
    def subtract(self,config,file, detector,*args,binsize = None,grid=[None,None,None], **kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        file = hdf5FileFixer(file)
        data_args = detector.split('[')
        y_stream_orig = detector
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        #Load Image for processing
        RIXS_Image = Object2dTransform()
        RIXS_Image.subtract(config,file,'Energy',detector,*args)
        #Load total 1D spectra for intensity comparison
        RIXS_Spectra = Load1d()
        RIXS_Spectra.subtract(config,file,'[None]',detector,*args)
        #Make another image, but use it for normalization
        XES_Image = Object2dReduce()
        XES_Image.load(RIXS_Image,0,0)
        #Reference is the max of the 1D data
        max_old = RIXS_Spectra.data[0][0].y_stream.max()
        #Mx of  the interopolated image and compare for 1D max
        XES_Image.roi('x', roi = (None,None))
        max_new = XES_Image.data[0][0].y_stream.max()
        RIXS_Image = Object2dTransform()
        RIXS_Image.subtract(config,file,'Energy',detector,*args, binsize_y = binsize,grid_x=grid, **kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args1[0].split(':')
            x_min = float(min(data_args2))
            x_max = float(max(data_args2))
            RIXS_Image.transform('x-y', xlim = (x_min,x_max))
            Object2dReduce.load(self, RIXS_Image,0, 0)
            Object2dReduce.roi(self,'x', roi =  (x_min,x_max))
        else:
            RIXS_Image.transform('x-y')
            Object2dReduce.load(self, RIXS_Image,0, 0)
            Object2dReduce.roi(self,'x', roi =  (None,None))
        scale_fact = max_old/max_new
        self.data[len(self.data)-1][0].y_stream = self.data[len(self.data)-1][0].y_stream*scale_fact
        self.data[len(self.data)-1][0].legend = str(len(self.data))+'-S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '-' + str(args[i])
        self.data[len(self.data)-1][0].legend += '_Energy_' + str(y_stream_orig)
    def compare(self,plot_object):
        """
        Loads data previously specified in a loader

        Parameters
        ----------
        plot_object: object
            name of the Loader object
        """
        for i in range(len(plot_object.data)):
            Load1d.loadObj(self, plot_object, i)
    def plot(self, linewidth = 2, xlabel='Energy Loss [eV]', ylabel='Relative Intensity', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        linewidth : int, optional
        title : string, optional
        xlabel : string, optional
        ylabel : string, optional
        ylabel_right : string, optional
        plot_height : int, optional
        plot_width : int, optional
        norm: boolean, optional
            Normalized plot output to [0,1]
        waterfall: float
            Normalizes plot output to [0,1] and applies offset specified
        kwargs
            all bokeh figure key-word arguments
        """
        Load1d.plot(self, linewidth = linewidth, xlabel=xlabel, ylabel=ylabel, plot_width = plot_width, plot_height = plot_height,**kwargs)
class ELOSSMapper(Object2dTransform):
    """ Load and plot x-ray emission map on energy loss scale """
    def load(self,config,filename,detector,*args,**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            REIXS beamline endstation configuration
        filename: string
            filename
        detector: string
            alias of the MCA detector
        arg: int
            scan number
        **kwargs
            norm: boolean
                Can be boolean or None (as False)
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize: int
                puts data in bins of specified size in the vertical direction
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        Object2dTransform.load(self,config,filename,'Energy',detector,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            x_range = (float(data_args3[0]), float(data_args3[1]))
            Object2dTransform.transform(self,'x-y', xlim = x_range)
        else:
            Object2dTransform.transform(self,'x-y')
        Object2dTransform.transpose(self)
    def stitch(self,config,filename,detector,*args,**kwargs):
        """
        Stitch specified scans for selected image.

        Parameters
        ----------
        See loader function.
        Stitches all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        Object2dTransform.stitch(self,config,filename,'Energy',detector,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            x_range = (float(data_args3[0]), float(data_args3[1]))
            Object2dTransform.transform(self,'x-y', xlim = x_range)
        else:
            Object2dTransform.transform(self,'x-y')
        Object2dTransform.transpose(self)
    def add(self,config,filename,detector,*args,**kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        Object2dTransform.add(self,config,filename,'Energy',detector,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            x_range = (float(data_args3[0]), float(data_args3[1]))
            Object2dTransform.transform(self,'x-y', xlim = x_range)
        else:
            Object2dTransform.transform(self,'x-y')
        Object2dTransform.transpose(self)
    def subtract(self,config,filename,detector,*args,**kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """
        filename = hdf5FileFixer(filename)
        data_args = detector.split('[')
        if(len(data_args) > 1):
            data_args1 = data_args[1].split(']')
            detector = data_args[0] + data_args1[1]
        else:
            detector = data_args[0]
        Object2dTransform.subtract(self,config,filename,'Energy',detector,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            x_range = (float(data_args3[0]), float(data_args3[1]))
            Object2dTransform.transform(self,'x-y', xlim = x_range)
        else:
            Object2dTransform.transform(self,'x-y')
        Object2dTransform.transpose(self)
    def plot(self, xlabel='Energy Loss [eV]', ylabel='Excitation Energy [eV]', plot_width = 900, plot_height = 600, **kwargs):
        """
        Plot all data assosciated with class instance/object.

        Parameters
        ----------
        title : string, optional
        kind : string, optional
        xlabel : string, optional
        ylabel : string, optional
        zlabel : string, optional
        plot_height : int, optional
        plot_width : int, optional
        vmin : float, optional
        vmax : float, optional
        colormap : string
            Use: "linear" or "log"
        norm : boolean
            to normalize the plot to the maximum
        kwargs
            all bokeh figure key-word arguments
        """
        Load2d.plot(self, xlabel=xlabel, ylabel=ylabel, plot_width = plot_width, plot_height = plot_height,**kwargs)    
    
    
class MCPLoader(Load2d):
    def load(self,config,file,detector,arg,**kwargs):
        """
        Load one or multiple specific scan(s) for selected streams.

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            filename
        detector: string
            alias of the MCP detector
        arg: int
            scan number
        **kwargs
            norm: boolean
                Can be boolean or None (as False)
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize: int
                puts data in bins of specified size in the vertical direction
        """
        file = hdf5FileFixer(file)
        detector = detector + '[None:None,None:None]'
        return Load2d.load(self,config,file,'[None]',detector,arg,**kwargs)
    def add(self,config,file,detector,arg,**kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        file = hdf5FileFixer(file)
        detector = detector + '[None:None,None:None]'
        return Load2d.add(self,config,file,'[None]',detector,arg,**kwargs)
    def add(self,config,file,detector,*args,**kwargs):
        """
        Add specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Adds all scans specified in *args.
        """
        file = hdf5FileFixer(file)
        detector = detector + '[None:None,None:None]'
        return Load2d.add(self,config,file,'[None]',detector,*args,**kwargs)
    def subtract(self,config,file,detector,*args,**kwargs):
        """
        Subtract specified images for selected streams.

        Parameters
        ----------
        See loader function.
        Subtracts all imnages from the first element.

        """
        file = hdf5FileFixer(file)
        detector = detector + '[None:None,None:None]'
        return Load2d.subtract(self,config,file,'[None]',detector,*args,**kwargs)
    def background_2d(self,config,file,detector,*args,**kwargs):
        """ Subtracts the defined data from all loaded data

        Parameters
        ----------
        config: dict
            h5 configuration
        file: string
            file name
        detector: string
            alias of the MCP detector
        *args: int
            scans
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
            binsize_x: int
                puts data in bins of specified size in the horizontal direction
            binsize: int
                puts data in bins of specified size in the vertical direction
        """
        file = hdf5FileFixer(file)
        detector = detector + '[None:None,None:None]'
        return Load2d.background_2d(self,config,file,'[None]',detector,*args,**kwargs)
    

#RSXS ES Configuration#
RSXS = h5Config()
RSXS.sca_folder('Data')
RSXS.key("SCAN_{scan:03d}",'scan')

#Aliases
RSXS.sca('Energy','Data/beam')
RSXS.sca('TEY','Data/tey')
RSXS.sca('I0','Data/i0')
RSXS.sca('TEY_N','Data/tey', norm_by = 'Data/i0')

#MCA Detectors
RSXS.mca('SDDX','Data/sdd_xrf_mca','Data/sdd_xrf_scale',None)
RSXS.mca('SDDA','Data/sdd_arm_mca','Data/sdd_arm_scale',None)
RSXS.mca('SDDX_N','Data/sdd_xrf_mca','Data/sdd_xrf_scale','Data/i0')
RSXS.mca('SDDA_N','Data/sdd_arm_mca','Data/sdd_arm_scale','Data/i0')

#IMAGE Detectors
RSXS.stack('mcpIMG','Data/mcp_a_img','Data/mcp_tth_scale','Data/mcp_detz_scale',None)
RSXS.stack('mcpIMG_N','Data/mcp_a_img',None,None,None)



#Log Book for RSXS ES
rsxs_log = dict()
rsxs_log['Command'] = 'command'
#rsxs_log['Sample'] = 'Endstation/Sample/Name'
rsxs_log['Comments'] = ('comment_01','comment_02','comment_03','comment_04','comment_05',
                        'comment_06','comment_07','comment_08','comment_09','comment_10')
rsxs_log['X'] = ['Endstation/Motors/x', 3]
rsxs_log['Y'] = ['Endstation/Motors/y', 3]
rsxs_log['Z'] = ['Endstation/Motors/z', 3]
rsxs_log['Theta'] = ['Endstation/Motors/th', 3]
rsxs_log['2Theta'] = ['Endstation/Motors/tth', 3]
rsxs_log['Chi'] = ['Endstation/Motors/chi',3]
rsxs_log['Phi'] = ['Endstation/Motors/phi',3]
rsxs_log['Detz'] = ['Endstation/Motors/detz',3]
rsxs_log['H'] = ['Endstation/Motors/H', 4]
rsxs_log['K'] = ['Endstation/Motors/K', 4]
rsxs_log['L'] = ['Endstation/Motors/L', 4]
rsxs_log['Temperature'] = ['Endstation/Counters/t_k', 2]
rsxs_log['Energy'] = ['Beamline/Monochromator/beam',2]
rsxs_log['Exit Slit'] = ['Beamline/Apertures/Exit_Slit/vert_gap',1]
rsxs_log['Flux'] = 'Beamline/flux'
rsxs_log['Dwell'] = ['Endstation/Counters/sec', 1]
rsxs_log['Mirror/Grating'] = ('/Beamline/Monochromator/grating',
                              '/Beamline/Monochromator/mirror')
rsxs_log['Polar/Harmonic'] = ('Beamline/Source/EPU/polarization', 
                              'Beamline/Source/EPU/harmonic')
rsxs_log['Status'] = 'status'
rsxs_log['Date'] = 'date'

#Data Structure Config
RIXS = h5Config()
RIXS.sca_folder('Data')
RIXS.key("SCAN_{scan:03d}",'scan')

#Aliases
RIXS.sca('Energy','Data/beam')
RIXS.sca('TEY','Data/tey')
RIXS.sca('I0','Data/i0')
RIXS.sca('TEY_N','Data/tey', norm_by = 'Data/i0')

#MCA Detectors
RIXS.mca('SDDA','Data/sdd_a_mca','Data/sdd_a_scale',None)
RIXS.mca('SDDA_NOSCALE','Data/sdd_a_mca',None,None)
RIXS.mca('SDDA_N','Data/sdd_a_mca','Data/sdd_a_scale',norm_by = 'Data/i0')
RIXS.mca('SDDB','Data/sdd_b_mca','Data/sdd_b_scale',None)
RIXS.mca('SDDB_NOSCALE','Data/sdd_b_mca',None,None)
RIXS.mca('SDDB_N','Data/sdd_b_mca','Data/sdd_b_scale',norm_by = 'Data/i0')
RIXS.mca('XEOL','Data/xeol_a_mca_norm','Data/xeol_a_scale',None)
RIXS.mca('XEOL_N','Data/xeol_a_mca_norm','Data/xeol_a_scale',norm_by = 'Data/i0')
RIXS.mca('XES','Data/mcp_xes_mca','Data/mcp_xes_scale',None)
RIXS.mca('XES_N','Data/mcp_xes_mca','Data/mcp_xes_scale',norm_by = 'Data/i0')

#IMAGE Detectors
RIXS.stack('mcpIMG_A','Data/mcp_a_img',None,None,None)
RIXS.stack('mcpIMG_B','Data/mcp_b_img',None,None,None)

#Log Book for RIXS ES
rixs_log = dict()

rixs_log['Command'] = 'command'
rixs_log['Sample'] = 'Endstation/Sample/Name'
rixs_log['Comments'] = ('comment_01','comment_02','comment_03','comment_04','comment_05',
                        'comment_06','comment_07','comment_08','comment_09','comment_10')
rixs_log['Horz (ssh)'] = ['Endstation/Motors/ssh',2]
rixs_log['Vert (ssv)'] = ['Endstation/Motors/ssv',2]
rixs_log['Depth (ssd)'] = ['Endstation/Motors/ssd',2]
rixs_log['Angle (ssa)'] = ['Endstation/Motors/ssa',1]
rixs_log['Temperature'] = ['Endstation/Counters/temp', 1]
rixs_log['Energy'] = ['Beamline/Monochromator/beam',2]
rixs_log['Exit Slit'] = ['Beamline/Apertures/Exit_Slit/vert_gap',1]
rixs_log['Flux'] = 'Beamline/flux'
rixs_log['Dwell'] = ['Endstation/Counters/sec', 1]
rixs_log['Mirror/Grating'] = ('/Beamline/Monochromator/grating',
                              '/Beamline/Monochromator/mirror')
rixs_log['Polar/Harmonic'] = ('Beamline/Source/EPU/polarization', 
                              'Beamline/Source/EPU/harmonic')
rixs_log['XES Energy'] = ['Endstation/Detectors/XES/mcp_mca_xes_energy', 2]
rixs_log['XES Grating'] = 'Endstation/Detectors/XES/mcp_mca_xes_grating'
rixs_log['XES Offset'] = ['Endstation/Detectors/XES/mcp_mca_xes_offset', 1]
rixs_log['Shift File'] = 'Endstation/Detectors/XES/mcp_mca_xes_shift_file'
rixs_log['XEOL Rate'] = ['Endstation/Detectors/XEOL/xeol_time_rate_a', 3]
rixs_log['Status'] = 'status'
rixs_log['Date'] = 'date'

