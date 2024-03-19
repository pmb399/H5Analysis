# Scientific modules
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from shapely.geometry import Point, Polygon
import skimage as ski
import lmfit
from lmfit.models import GaussianModel, QuadraticModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel, LorentzianModel, ExponentialModel

# Import Loaders
from .LoadData import Load1d, Load2d, LoadHistogram

# Import simplemath and datautil
from .simplemath import grid_data_mesh, handle_eval, grid_data, apply_offset, apply_savgol, bin_data
from .datautil import mca_roi, get_indices, get_indices_polygon

class Object1dAddSubtract(Load1d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()
        self.DataObjectsSubtract = list()
        self.x_string = ""
        self.y_string = ""
        self.scan_string = "S"

        return Load1d.__init__(self)    
        
    def add(self,obj,line,scan):
        """Loader objects to be added
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """

        o = obj.data[line][scan]
        self.x_string += f"{o.xlabel}|"
        self.y_string += f"{o.ylabel}|"
        self.scan_string += f"_+{scan}"

        self.DataObjectsAdd.append(o)
        
    def subtract(self,obj,line,scan):
        """Loader objects to be subtracted
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """

        o = obj.data[line][scan]
        self.x_string += f"{o.xlabel}|"
        self.y_string += f"{o.ylabel}|"
        self.scan_string += f"_-{scan}"

        self.DataObjectsSubtract.append(o)
        
    def evaluate(self):
        """Evaluate the request"""

        if self.DataObjectsAdd == []:
            raise Exception('You need to add at least one scan.')

        all_objs = self.DataObjectsAdd + self.DataObjectsSubtract
        start_list = list()
        end_list = list()

        for i,item in enumerate(all_objs):
            start_list.append(item.x_stream.min())
            end_list.append(item.x_stream.max())

            if i == 0:
                x_diff = np.abs(np.diff(item.x_stream).min())

        s = max(start_list)
        e = min(end_list)

        if s>=e:
            raise Exception("There is not sufficient overlap in x to perform interpolation.")
        
        # Limit array size to 100MB (=104857600 bytes)
        # Numpy float64 array element requires 8 bytes
        max_steps = 104857600/8
        steps = int((e-s)/x_diff)+1

        if steps>max_steps:
            num = max_steps
        else:
            num = steps

        MASTER_x = np.linspace(s,e,num)

        # First, add all objects
        for i,item in enumerate(self.DataObjectsAdd):
            # Determine MASTER streams
            if i ==0:
                MASTER_y = interp1d(item.x_stream,item.y_stream)(MASTER_x)
            
            # Interpolate other data and add
            else:
                MASTER_y += interp1d(item.x_stream,item.y_stream)(MASTER_x)
        
        # Second, subtract objects
        for i,item in enumerate(self.DataObjectsSubtract):
                MASTER_y -= interp1d(item.x_stream,item.y_stream)(MASTER_x)

        # Store data
        class added_object:
            def __init__(self):
                pass
        
        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()
        data[0].x_stream = MASTER_x
        data[0].y_stream = MASTER_y
        data[0].scan = self.scan_string
        index = len(self.data) + 1
        data[0].legend = f'{index} - {self.scan_string} - Addition/Subtraction'

        data[0].xlabel = self.x_string
        data[0].ylabel = self.y_string
        data[0].filename = 'Object Math'
        
        self.data.append(data)

#########################################################################################

class Object1dStitch(Load1d):
    """Apply stitching on loader objects"""

    def __init__(self):
        self.DataObjectsStitch = list()
        self.x_string = ""
        self.y_string = ""
        self.scan_string = "S"

        return Load1d.__init__(self)
        
    def stitch(self,obj,line,scan):
        """Loader objects to be added
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """

        o = obj.data[line][scan]
        self.x_string += f"{o.xlabel}|"
        self.y_string += f"{o.ylabel}|"
        self.scan_string += f"_{scan}"

        self.DataObjectsStitch.append(o)
                
    def evaluate(self):
        """Evaluate the request"""

        if self.DataObjectsStitch == []:
            raise Exception('You need to add at least one scan.')

        start_list = list()
        end_list = list()
        diff_list = list()

        for i,item in enumerate(self.DataObjectsStitch):
            start_list.append(item.x_stream.min())
            end_list.append(item.x_stream.max())
            diff_list.append(np.abs(np.diff(item.x_stream).min()))

        s = min(start_list)
        e = max(end_list)
        x_diff = min(diff_list)
        
        # Limit array size to 100MB (=104857600 bytes)
        # Numpy float64 array element requires 8 bytes
        max_steps = 104857600/8
        steps = int((e-s)/x_diff)+1

        if steps>max_steps:
            num = max_steps
        else:
            num = steps

        MASTER_x = np.linspace(s,e,num)

        # Store all y values
        MASTER_y_list = list() # y-arrays interpolated to common scale
        MASTER_y_nan_list = list() # where nan values are stored

        # Iterate over all loaded scans for interpolation
        for i, item in enumerate(self.DataObjectsStitch):
            # interpolate to common scale
            item = interp1d(item.x_stream,item.y_stream,bounds_error=False)(MASTER_x)
            # Store results
            MASTER_y_list.append(item)
            # Return boolean True where array element is a number
            MASTER_y_nan_list.append(~np.isnan(item))

        # This is for averaging
        # Sum the arrays of common length, treat nan as 0
        # For each element, sum how many True (numbers) contribute to the sum
        # Normalize to get average by array division
        MASTER_y = np.nansum(MASTER_y_list,axis=0)/np.sum(MASTER_y_nan_list,axis=0)

        # Store data
        class added_object:
            def __init__(self):
                pass
        
        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()
        data[0].x_stream = MASTER_x
        data[0].y_stream = MASTER_y
        data[0].scan = self.scan_string
        index = len(self.data) + 1
        data[0].legend = f'{index} - {self.scan_string} - Stitching'

        data[0].xlabel = self.x_string
        data[0].ylabel = self.y_string
        data[0].filename = 'Object Math'
        
        self.data.append(data)

#########################################################################################


class Object2dAddSubtract(Load2d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()
        self.DataObjectsSubtract = list()
        self.x_string = ""
        self.y_string = ""
        self.z_string = ""
        self.scan_string = "S"

        return Load2d.__init__(self)
    
    def load(self):
        raise Exception("This method is not defined")
        
    def add(self,obj,line,scan):
        """Loader objects to be added
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """

        o = obj.data[line][scan]
        self.x_string += f"{o.xlabel}|"
        self.y_string += f"{o.ylabel}|"
        self.z_string += f"{o.zlabel}|"
        self.scan_string += f"_+{scan}"

        self.DataObjectsAdd.append(o)
        
    def subtract(self,obj,line,scan):
        """Loader objects to be subtracted
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """

        o = obj.data[line][scan]
        self.x_string += f"{o.xlabel}|"
        self.y_string += f"{o.ylabel}|"
        self.z_string += f"{o.zlabel}|"
        self.scan_string += f"_-{scan}"

        self.DataObjectsSubtract.append(o)
        
    def evaluate(self):
        """Evaluate the request"""

        # Make sure there is no other scan loaded
        if self.data != []:
            raise UserWarning("Can only load one scan at a time.")
        
        if self.DataObjectsAdd == []:
            raise Exception('You need to add at least one scan.')
        
        all_objs = self.DataObjectsAdd + self.DataObjectsSubtract
        x_start_list = list()
        x_end_list = list()
        y_start_list = list()
        y_end_list = list()
        x_list = list()
        y_list = list()

        for i,item in enumerate(all_objs):
            x_start_list.append(item.new_x.min())
            x_end_list.append(item.new_x.max())
            y_start_list.append(item.new_y.min())
            y_end_list.append(item.new_y.max())
            x_list.append(item.new_x)
            y_list.append(item.new_y)

            if i == 0:
                x_comp = item.new_x
                y_comp = item.new_y
                x_diff = np.abs(np.diff(item.new_x).min())
                y_diff = np.abs(np.diff(item.new_y).min())

        #Check if we need to interpolate scales
        if all([np.array_equal(x_comp,test_scale) for test_scale in x_list]) and all([np.array_equal(y_comp,test_scale) for test_scale in y_list]):
            # All scales are equal, no interpolation required
            MASTER_x_stream = x_comp
            MASTER_y_stream = y_comp

            for i,item in enumerate(self.DataObjectsAdd):
                if i ==0:
                    MASTER_detector = item.new_z
                    
            # Add all objects (2d) after interpolation step to master
                else:
                    MASTER_detector = np.add(MASTER_detector,item.new_z)
            
            # Add all objects (2d) that need to be removed after interpolation step to master
            for i,item in enumerate(self.DataObjectsSubtract):
                    if i == 0:
                        SUB_detector = item.new_z
                    else:
                        SUB_detector = np.add(SUB_detector,item.new_z)

            # Remove subtraction from Master, if any
            if len(self.DataObjectsSubtract)>0:
                MASTER_detector = np.subtract(MASTER_detector,SUB_detector)

        else:
            # Interpolation required
            x_s = max(x_start_list)
            x_e = min(x_end_list)
            y_s = max(y_start_list)
            y_e = min(y_end_list)

            if x_s>=x_e:
                raise Exception("There is not sufficient overlap in x to perform interpolation.")
            
            if y_s>=y_e:
                raise Exception("There is not sufficient overlap in y to perform interpolation.")
            
            # Limit array size to 100MB (=104857600 bytes)
            # Numpy float64 array element requires 8 bytes
            max_steps = 104857600/8
            x_steps = int((x_e-x_s)/x_diff)+1
            y_steps = int((y_e-y_s)/y_diff)+1

            if x_steps*y_steps>max_steps:
                step_norm = int(np.ceil(np.sqrt(x_steps*y_steps/13107200)))
                x_num = int(x_steps/step_norm)
                y_num = int(y_steps/step_norm)
            else:
                x_num = x_steps
                y_num = y_steps

            MASTER_x_stream = np.linspace(x_s,x_e,x_num)
            MASTER_y_stream = np.linspace(y_s,y_e,y_num)


            # Set up the new master streams
            for i,item in enumerate(self.DataObjectsAdd):
                if i ==0:
                    MASTER_detector = interp2d(item.new_x,item.new_y,item.new_z)(MASTER_x_stream,MASTER_y_stream)
                    
            # Add all objects (2d) after interpolation step to master
                else:
                    interp = interp2d(item.new_x,item.new_y,item.new_z)
                    new_z = interp(MASTER_x_stream,MASTER_y_stream)

                    MASTER_detector = np.add(MASTER_detector,new_z)
            
            # Add all objects (2d) that need to be removed after interpolation step to master
            for i,item in enumerate(self.DataObjectsSubtract):
                    interp = interp2d(item.new_x,item.new_y,item.new_z)
                    new_z = interp(MASTER_x_stream,MASTER_y_stream)

                    if i == 0:
                        SUB_detector = new_z
                    else:
                        SUB_detector = np.add(SUB_detector,new_z)

            # Remove subtraction from Master, if any
            if len(self.DataObjectsSubtract)>0:
                MASTER_detector = np.subtract(MASTER_detector,SUB_detector)
        
        # Store data
        class added_object:
            def __init__(self):
                pass
        
        data = dict()
        data[0] = added_object()
        data[0].new_x = MASTER_x_stream
        data[0].new_y = MASTER_y_stream
        data[0].new_z = MASTER_detector
        data[0].xmin = MASTER_x_stream.min()
        data[0].xmax = MASTER_x_stream.max()
        data[0].ymin = MASTER_y_stream.min()
        data[0].ymax = MASTER_y_stream.max()
        data[0].scan = self.scan_string
        index = len(self.data) + 1
        data[0].legend = f'{index} - {self.scan_string} - Addition/Subtraction'
        data[0].xlabel = self.x_string
        data[0].ylabel = self.y_string
        data[0].zlabel = self.z_string
        data[0].filename = 'Simple Math'
        
        self.data.append(data)

#########################################################################################

class Object2dReduce(Load1d):
    """Apply reduction from 2d data to 1d"""

    def load(self,obj,line,scan):
        """Loader for 2d object
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """
        self.MCADataObject = obj.data[line][scan]

    def add(self):
        raise Exception("This method is not defined")
    
    def subtract(self):
        raise Exception("This method is not defined")
    
    def stitch(self):
        raise Exception("This method is not defined")
    
    def background(self):
        raise Exception("This method is not defined")
    
    def loadObj(self):
        raise Exception("This method is not defined")

    def roi(self,integration_axis,roi=(None,None)):
        """ Apply an region of interest (ROI) to one axis
        
            Parameters
            ----------
            integration_axis: string
                options: 'x' or 'y'
            roi: tuple
                specify the lower and upper bounds, i.e. (lower,upper)
        """

        # Get the data from the loaded object
        x = self.MCADataObject.new_x
        y = self.MCADataObject.new_y
        z = self.MCADataObject.new_z

        # Prepare data storage
        class added_object:
            def __init__(self):
                pass
        
        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()

        # Integrate depending on chosen axis
        # Note, in matrix notation x axis (horizontal) corresponds to 1st axis
        if integration_axis == 'x':
            idx_low, idx_high = get_indices(roi,x)
            sum = mca_roi(z,idx_low,idx_high,1)
            MASTER_x = y
            data[0].xlabel = self.MCADataObject.ylabel

        elif integration_axis == 'y':
            idx_low, idx_high = get_indices(roi,y)
            sum = mca_roi(z,idx_low,idx_high,0)
            MASTER_x = x
            data[0].xlabel = self.MCADataObject.xlabel

        else:
            raise Exception('Specified integration axis not defined.')

        # Store all pertinent information in object
        data[0].x_stream = MASTER_x
        data[0].y_stream = sum
        data[0].scan = self.MCADataObject.scan
        data[0].ylabel = f"{self.MCADataObject.zlabel} - ROI"
        index = len(self.data) + 1
        data[0].legend = f'{index} - S{self.MCADataObject.scan} - 2d ROI reduction'
        data[0].filename = self.MCADataObject.filename
        
        self.data.append(data)


    def polygon(self,integration_axis,polygon,exact=False):
        """ Mask array defined by a polygon and sum over specified axis
        
            Parameters
            ----------
            integration_axis: string
                options: 'x' or 'y'
            polygon: list of tuple
                corresponds to coordinate pairs
            exact: Boolean
                True: Iterates over all data points P explicitly and determines if P is contained in polygon (most accurate, slow)
                False: Applies image algorithm to mask polygon by filling holes (less accurate, faster)
        """

        # Get the data from the loaded object
        x = self.MCADataObject.new_x
        y = self.MCADataObject.new_y
        z = self.MCADataObject.new_z

        # Prepare data storage
        class added_object:
            def __init__(self):
                pass
        
        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()

        # Choose integration axis
        if integration_axis == 'x':                
            if exact == False:
                # If not exact, 
                # return array indices corresponding to polygon boundaries
                # then apply algorithm to get mask
                # apply mask to array
                idx = get_indices_polygon(polygon,x,y)
                mask = ski.draw.polygon2mask(z.shape, idx)
                sum = np.where(mask,z,0).sum(axis=1)

            else:
                # If exact,
                # get shapely polygon
                # check iteratively for each data point in array if contained
                # in polygon.
                # if so, add to sum
                poly = Polygon(polygon)
                pXES = list()
                for y_idx,py in enumerate(y):
                    x_sum = 0
                    for x_idx,px in enumerate(x):
                        p = Point(px,py)
                        if poly.contains(p):
                            x_sum += z[y_idx,x_idx]
                    pXES.append(x_sum)

                sum = np.array(pXES)

            # Set labels and independent data stream
            data[0].xlabel = self.MCADataObject.ylabel
            MASTER_x = y

        elif integration_axis == 'y':
            if exact == False:
                idx = get_indices_polygon(polygon,x,y)
                mask = ski.draw.polygon2mask(z.shape, idx)
                sum = np.where(mask,z,0).sum(axis=0)

            else:
                poly = Polygon(polygon)
                pXES = list()
                for x_idx,px in enumerate(x):
                    y_sum = 0
                    for y_idx,py in enumerate(y):
                        p = Point(px,py)
                        if poly.contains(p):
                            y_sum += z[y_idx,x_idx]
                    pXES.append(y_sum)

                sum = np.array(pXES)

            data[0].xlabel = self.MCADataObject.xlabel                
            MASTER_x = x

        else:
            raise Exception('Specified integration axis not defined.')

        # Store all pertinent information in object
        data[0].x_stream = MASTER_x
        data[0].y_stream = sum
        data[0].scan = self.MCADataObject.scan
        data[0].ylabel = f"{self.MCADataObject.zlabel} - ROI"
        index = len(self.data) + 1 
        data[0].legend = f'{index} - S{self.MCADataObject.scan} - 2d polygon reduction'
        data[0].filename = self.MCADataObject.filename
        
        self.data.append(data)


    def apply_kwargs(self,norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None):
        """ Apply math to 1d reduced objects

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
                grid_x: list
                    grid data evenly with [start,stop,delta]
                savgol: tuple
                    (window length, polynomial order, derivative)
                binsize: int
                    puts data in bins of specified size
        """

        for i, val in enumerate(self.data):
            for k, v in val.items():

                #Bin the data if requested
                if binsize != None:
                    v.x_stream, v.y_stream = bin_data(v.x_stream,v.y_stream,binsize)

                # Grid the data if specified
                if grid_x != [None, None, None]:
                    new_x, new_y = grid_data(
                        v.x_stream, v.y_stream, grid_x)

                    v.x_stream = new_x
                    v.y_stream = new_y

                # Apply offsets to x-stream
                v.x_stream = apply_offset(
                v.x_stream, xoffset, xcoffset)

                # Apply normalization to [0,1]
                if norm == True:
                    v.y_stream = np.interp(
                        v.y_stream, (v.y_stream.min(), v.y_stream.max()), (0, 1))

                # Apply offset to y-stream
                v.y_stream = apply_offset(
                v.y_stream, yoffset, ycoffset)
                    
                # Smooth and take derivatives
                if savgol != None:
                    if isinstance(savgol,tuple):
                        if len(savgol) == 2: # Need to provide window length and polynomial order
                            savgol_deriv = 0 # Then, no derivative is taken
                        elif len(savgol) == 3:
                            savgol_deriv = savgol[2] # May also specify additional argument for derivative order
                        else:
                            raise TypeError("Savgol smoothing arguments incorrect.")
                        v.x_stream, v.y_stream = apply_savgol(v.x_stream,v.y_stream,savgol[0],savgol[1],savgol_deriv)

                        if norm == True:
                            v.y_stream = v.y_stream / \
                           v.y_stream.max()
                    else:
                        raise TypeError("Savgol smoothing arguments incorrect.")
                    
                self.data[i][k].x_stream = v.x_stream
                self.data[i][k].y_stream = v.y_stream

class Object2dTransform(Load2d):
    """Apply transformations to a 2d image"""

    def transform(self,trans_y,xlim=(None,None),ylim=(None,None)):
        """ Apply math operations on a per data point basis. Change second axis (y) for all data along first (x) axis

            Parameters
            ----------
            trans_y: string
                math expression to be evaluated at every x data point
                available variables include 'x', 'y', 'z'.
            xlim: tuple
                specify lower and upper bound of x-limits, cropped before transformation
            ylim: tuple
                specify lower and upper bound of y-limits included in matrix after transformation
        """

        # Do this for all scans in loaded object
        for i, val in enumerate(self.data):
            for k, v in val.items():

                # Get data as variables x, y, z
                y = v.new_y
                z = v.new_z

                # Restrict on xlim if requested
                if xlim[0] != None:
                    xlow = (np.abs(xlim[0]-v.new_x)).argmin()
                else:
                    xlow = None
                if xlim[1] != None:
                    xhigh = (np.abs(xlim[1]-v.new_x)).argmin()
                else:
                    xhigh = None

                # Adjust the scale and image
                v.new_x = v.new_x[xlow:xhigh]
                v.xmin = min(v.new_x)
                v.xmax = max(v.new_x)
                z = z[:,xlow:xhigh]

                # Store the axes modified for each data point
                axes_math = list()
                for x in v.new_x:
                    # For each data point, evaluate the math and store y-axis
                    axes_math.append(handle_eval(trans_y,locals()))

                # Find min/max values
                mins = list()
                maxs = list()
                for arr in axes_math:
                    mins.append(min(arr))
                    maxs.append(max(arr))
                
                # May also choose to limit y-range
                # Need the biggest common overlap or as requested
                if (ylim[0] == None and ylim[1] == None):
                    ymin = max(mins)
                    ymax = min(maxs)
                    yaxes = axes_math
                else:
                    ydict = dict()
                    # Check which scales contain lower and upper bound of y
                    for idx,arr in enumerate(axes_math):
                        if min(arr) <= ylim[0] <= max(arr):
                            if min(arr) <= ylim[1] <= max(arr):
                                ydict[idx] = arr

                    # Need to have at least one scale that contains both ymin and ymax
                    if len(ydict.keys()) == 0:
                        raise Exception('Invalid y-range specified.')
                    
                    # Set the y range and set available axes
                    ymin = ylim[0]
                    ymax = ylim[1]
                    yaxes = list(ydict.values())

                    # Extract indices to find min/max x-values supported
                    # And crop axis
                    # works because we iterate over x when checking the y-scales
                    xindices = list(ydict.keys())                        
                    v.new_x = v.new_x[xindices]
                    v.xmin = min(v.new_x)
                    v.xmax = max(v.new_x)

                    # Also crop down matrix accordingly
                    z = z[:,xindices]

                # Create new, common y scale
                new_y = np.linspace(ymin, ymax, len(v.new_y), endpoint=True)

                # Store shifted data in new array
                scatter_z = np.zeros((len(new_y),len(v.new_x)))

                # Evaluate the image on the new common energy axis
                for idx,val in enumerate(np.transpose(z)):
                    scatter_z[:,idx] = interp1d(yaxes[idx],val)(new_y)
                
                # Note, loading with Load2d ensures that the x-axis (v.new_x) is
                # is already interpolated on evenly spaced grid

                # Update data as calculated above
                v.new_z = scatter_z
                v.new_y = new_y
                v.ymin = ymin
                v.ymax = ymax
                v.ylabel = f'Transformed {v.ylabel}'

                self.data[i][k] = v

    def transpose(self):
        """ Transpose loaded image and swap axes """

        # For all loaded objects
        for i, val in enumerate(self.data):
            for k, v in val.items():

                # apply transpose to image matrix
                v.new_z = np.transpose(v.new_z)

                # update the axes and labels
                new_x = v.new_y
                new_y = v.new_x
                xmin = v.ymin
                xmax = v.ymax
                xlabel = v.ylabel
                ymin = v.xmin
                ymax = v.xmax
                ylabel = v.xlabel

                v.new_x = new_x
                v.new_y = new_y
                v.xmin = xmin
                v.xmax = xmax
                v.xlabel = xlabel
                v.ymin = ymin
                v.ymax = ymax
                v.ylabel = ylabel

                # Write back to dict
                self.data[i][k] = v

#########################################################################################
                
class Object1dFit(Load1d):
    """Apply fit to 1d data"""

    def load(self,obj,line,scan):
        """Loader for 2d object
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """

        self.SCADataObject = obj.data[line][scan]
        data_dict = dict()
        data_dict[scan] = self.SCADataObject
        self.data.append(data_dict)
        self.fitComponents = dict()
        self.fitSpectra = dict()

    def add(self):
        raise Exception("This method is not defined")
    
    def subtract(self):
        raise Exception("This method is not defined")
    
    def stitch(self):
        raise Exception("This method is not defined")
    
    def background(self):
        raise Exception("This method is not defined")
    
    def loadObj(self):
        raise Exception("This method is not defined")
    
    def add_Gaussian(self,center,amplitude,sigma,center_bounds=(None,None),center_vary=True,amplitude_bounds=(None,None),amplitude_vary=True,sigma_bounds=(None,None),sigma_vary=True):
        """Add Gaussian LMFit model
        
            Parameters
            ----------
            center: float
                Center position of the Gaussian
            amplitude: float
                Amplitude of the Gaussian
            sigma: float
                The standard deviation of the Gaussian, note that FWHM = 2.355 * sigma
            center_bounds: tuple
                Specify the lower and upper bounds of the parameter
            center_vary: Boolean
                Specify whether the paramter is being fit
            amplitude_bounds: tuple
                Specify the lower and upper bounds of the parameter
            amplitude_vary: Boolean
                Specify whether the paramter is being fit
            sigma_bounds: tuple
                Specify the lower and upper bounds of the parameter
            sigma_vary: Boolean
                Specify whether the paramter is being fit
        """
        
        parameters = dict()

        # Center
        pars = dict()
        pars['value'] = center
        if center_bounds[0] != None:
            pars['min'] = center_bounds[0]
        if center_bounds[1] != None:
            pars['max'] = center_bounds[1]
        pars['vary'] = center_vary
        parameters['center'] = pars

        # amplitude
        pars = dict()
        pars['value'] = amplitude
        if amplitude_bounds[0] != None:
            pars['min'] = amplitude_bounds[0]
        if amplitude_bounds[1] != None:
            pars['max'] = amplitude_bounds[1]
        pars['vary'] = amplitude_vary
        parameters['amplitude'] = pars

        # sigma
        pars = dict()
        pars['value'] = sigma
        if sigma_bounds[0] != None:
            pars['min'] = sigma_bounds[0]
        if sigma_bounds[1] != None:
            pars['max'] = sigma_bounds[1]
        pars['vary'] = sigma_vary
        parameters['sigma'] = pars

        parameters['model'] = GaussianModel
        self.fitComponents[f"g{len(list(self.fitComponents))}_"] = parameters


    def add_Constant(self, constant, constant_bounds=(None,None), constant_vary=True):

        """Add Constant LMFit model
        
            Parameters
            ----------
            constant: float
                y-value of the constant
            constant_bounds: tuple
                Specify the lower and upper bounds of the parameter
            constant_vary: Boolean
                Specify whether the paramter is being fit
        """

        parameters = dict()

        # constant
        pars = dict()
        pars['value'] = constant
        if constant_bounds[0] != None:
            pars['min'] = constant_bounds[0]
        if constant_bounds[1] != None:
            pars['max'] = constant_bounds[1]
        pars['vary'] = constant_vary
        parameters['constant'] = pars

        parameters['model'] = ConstantModel
        self.fitComponents[f"c{len(list(self.fitComponents))}_"] = parameters
        

    def add_Linear(self,slope,intercept,slope_bounds=(None,None),slope_vary=True,intercept_bounds=(None,None),intercept_vary=True):
        """Add Linear LMFit model
        
            Parameters
            ----------
            slope: float
                Slope of the linear function (m in f(x) = m*x + b)
            intercept: float
                y-intercept  (b in f(x) = m*x + b)
            slope_bounds: tuple
                Specify the lower and upper bounds of the parameter
            slope_vary: Boolean
                Specify whether the paramter is being fit
            intercept_bounds: tuple
                Specify the lower and upper bounds of the parameter
            intercept_vary: Boolean
                Specify whether the paramter is being fit
        """

        parameters = dict()

        # slope
        pars = dict()
        pars['value'] = slope
        if slope_bounds[0] != None:
            pars['min'] = slope_bounds[0]
        if slope_bounds[1] != None:
            pars['max'] = slope_bounds[1]
        pars['vary'] = slope_vary
        parameters['slope'] = pars

        # intercept
        pars = dict()
        pars['value'] = intercept
        if intercept_bounds[0] != None:
            pars['min'] = intercept_bounds[0]
        if intercept_bounds[1] != None:
            pars['max'] = intercept_bounds[1]
        pars['vary'] = intercept_vary
        parameters['intercept'] = pars

        parameters['model'] = LinearModel
        self.fitComponents[f"lin{len(list(self.fitComponents))}_"] = parameters

    def add_Quadratic(self,a,b,c,a_bounds=(None,None),a_vary=True,b_bounds=(None,None),b_vary=True,c_bounds=(None,None),c_vary=True):
        """Add Quadratic LMFit model
        
            Parameters
            ----------
            a: float
                Parameter a in f(x) = a*x^2 + b*x + c
            b: float
                Parameter b in f(x) = a*x^2 + b*x + c
            c: float
                Parameter c in f(x) = a*x^2 + b*x + c
            a_bounds: tuple
                Specify the lower and upper bounds of the parameter
            a_vary: Boolean
                Specify whether the paramter is being fit
            b_bounds: tuple
                Specify the lower and upper bounds of the parameter
            b_vary: Boolean
                Specify whether the paramter is being fit
            c_bounds: tuple
                Specify the lower and upper bounds of the parameter
            c_vary: Boolean
                Specify whether the paramter is being fit
        """

        parameters = dict()

        # a
        pars = dict()
        pars['value'] = a
        if a_bounds[0] != None:
            pars['min'] = a_bounds[0]
        if a_bounds[1] != None:
            pars['max'] = a_bounds[1]
        pars['vary'] = a_vary
        parameters['a'] = pars

        # b
        pars = dict()
        pars['value'] = b
        if b_bounds[0] != None:
            pars['min'] = b_bounds[0]
        if b_bounds[1] != None:
            pars['max'] = b_bounds[1]
        pars['vary'] = b_vary
        parameters['b'] = pars

        # c
        pars = dict()
        pars['value'] = c
        if c_bounds[0] != None:
            pars['min'] = c_bounds[0]
        if c_bounds[1] != None:
            pars['max'] = c_bounds[1]
        pars['vary'] = c_vary
        parameters['c'] = pars

        parameters['model'] = QuadraticModel
        self.fitComponents[f"q{len(list(self.fitComponents))}_"] = parameters

    def add_Polynomial(self,c1,c2,c3,c4,c5,c6,c7,c1_bounds=(None,None),c1_vary=True,c2_bounds=(None,None),c2_vary=True,c3_bounds=(None,None),c3_vary=True,c4_bounds=(None,None),c4_vary=True,c5_bounds=(None,None),c5_vary=True,c6_bounds=(None,None),c6_vary=True,c7_bounds=(None,None),c7_vary=True):
        """Add Polynomial LMFit model
        
            Parameters
            ----------
            For 1<=i<=7
            ci: float
                Parameter in f(x) = sum c_i*x^i
            ci_bounds: tuple
                Specify the lower and upper bounds of the parameter
            ci_vary: Boolean
                Specify whether the paramter is being fit
        """
        
        parameters = dict()

        # c1
        pars = dict()
        pars['value'] = c1
        if c1_bounds[0] != None:
            pars['min'] = c1_bounds[0]
        if c1_bounds[1] != None:
            pars['max'] = c1_bounds[1]
        pars['vary'] = c1_vary
        parameters['c1'] = pars

        # c2
        pars = dict()
        pars['value'] = c2
        if c2_bounds[0] != None:
            pars['min'] = c2_bounds[0]
        if c2_bounds[1] != None:
            pars['max'] = c2_bounds[1]
        pars['vary'] = c2_vary
        parameters['bc2'] = pars

        # c3
        pars = dict()
        pars['value'] = c3
        if c3_bounds[0] != None:
            pars['min'] = c3_bounds[0]
        if c3_bounds[1] != None:
            pars['max'] = c3_bounds[1]
        pars['vary'] = c3_vary
        parameters['c3'] = pars

        # c4
        pars = dict()
        pars['value'] = c4
        if c4_bounds[0] != None:
            pars['min'] = c4_bounds[0]
        if c4_bounds[1] != None:
            pars['max'] = c4_bounds[1]
        pars['vary'] = c4_vary
        parameters['c4'] = pars

        # c5
        pars = dict()
        pars['value'] = c5
        if c5_bounds[0] != None:
            pars['min'] = c5_bounds[0]
        if c5_bounds[1] != None:
            pars['max'] = c5_bounds[1]
        pars['vary'] = c5_vary
        parameters['c5'] = pars

        # c6
        pars = dict()
        pars['value'] = c6
        if c6_bounds[0] != None:
            pars['min'] = c6_bounds[0]
        if c6_bounds[1] != None:
            pars['max'] = c6_bounds[1]
        pars['vary'] = c6_vary
        parameters['c6'] = pars

        # c7
        pars = dict()
        pars['value'] = c7
        if c7_bounds[0] != None:
            pars['min'] = c7_bounds[0]
        if c7_bounds[1] != None:
            pars['max'] = c7_bounds[1]
        pars['vary'] = c7_vary
        parameters['c7'] = pars

        parameters['model'] = PolynomialModel
        self.fitComponents[f"p{len(list(self.fitComponents))}_"] = parameters


    def add_Exponential(self,decay,amplitude,decay_bounds=(None,None),decay_vary=True,amplitude_bounds=(None,None),amplitude_vary=True):
        """Add Exponential LMFit model
        
            Parameters
            ----------
            decay: float
                Decay parameter lambda in f(x) = A*e^(-x/lambda)
            amplitude: float
                Amplitude paramter A in f(x) = A*e^(-x/lambda)
            decay_bounds: tuple
                Specify the lower and upper bounds of the parameter
            decay_vary: Boolean
                Specify whether the paramter is being fit
            amplitude_bounds: tuple
                Specify the lower and upper bounds of the parameter
            amplitude_vary: Boolean
                Specify whether the paramter is being fit
        """

        parameters = dict()

        # decay
        pars = dict()
        pars['value'] = decay
        if decay_bounds[0] != None:
            pars['min'] = decay_bounds[0]
        if decay_bounds[1] != None:
            pars['max'] = decay_bounds[1]
        pars['vary'] = decay_vary
        parameters['decay'] = pars

        # amplitude
        pars = dict()
        pars['value'] = amplitude
        if amplitude_bounds[0] != None:
            pars['min'] = amplitude_bounds[0]
        if amplitude_bounds[1] != None:
            pars['max'] = amplitude_bounds[1]
        pars['vary'] = amplitude_vary
        parameters['amplitude'] = pars

        parameters['model'] = ExponentialModel
        self.fitComponents[f"e{len(list(self.fitComponents))}_"] = parameters


    def add_Lorentzian(self,center,amplitude,sigma,center_bounds=(None,None),center_vary=True,amplitude_bounds=(None,None),amplitude_vary=True,sigma_bounds=(None,None),sigma_vary=True):
        """Add Lorentzian LMFit model
        
            Parameters
            ----------
            center: float
                Center position of the Gaussian
            amplitude: float
                Amplitude of the Gaussian
            sigma: float
                The standard deviation of the Gaussian, note that FWHM = 2.355 * sigma
            center_bounds: tuple
                Specify the lower and upper bounds of the parameter
            center_vary: Boolean
                Specify whether the paramter is being fit
            amplitude_bounds: tuple
                Specify the lower and upper bounds of the parameter
            amplitude_vary: Boolean
                Specify whether the paramter is being fit
            sigma_bounds: tuple
                Specify the lower and upper bounds of the parameter
            sigma_vary: Boolean
                Specify whether the paramter is being fit
        """
        
        parameters = dict()

        # Center
        pars = dict()
        pars['value'] = center
        if center_bounds[0] != None:
            pars['min'] = center_bounds[0]
        if center_bounds[1] != None:
            pars['max'] = center_bounds[1]
        pars['vary'] = center_vary
        parameters['center'] = pars

        # amplitude
        pars = dict()
        pars['value'] = amplitude
        if amplitude_bounds[0] != None:
            pars['min'] = amplitude_bounds[0]
        if amplitude_bounds[1] != None:
            pars['max'] = amplitude_bounds[1]
        pars['vary'] = amplitude_vary
        parameters['amplitude'] = pars

        # sigma
        pars = dict()
        pars['value'] = sigma
        if sigma_bounds[0] != None:
            pars['min'] = sigma_bounds[0]
        if sigma_bounds[1] != None:
            pars['max'] = sigma_bounds[1]
        pars['vary'] = sigma_vary
        parameters['sigma'] = pars

        parameters['model'] = LorentzianModel
        self.fitComponents[f"lo{len(list(self.fitComponents))}_"] = parameters

    def evaluate(self,lower_limit=None,upper_limit=None,fit='best'):
        """Construct and evaluate composite LMFit model
        
            Parameters
            ----------
            kwargs:
                lower_limit: float, None
                    Lower boundary for the minimizer evaluation, ignored if set to None
                upper_limit: float, None
                    Upper boundary for the minimizer evaluation, ignored if set to None
                fit: string
                    Options:
                        * 'best' - displays the best fit
                        * 'init' - displays the initial components
                        * 'components' - displays the best fit with the optimized components
        """
        
        # Construct composite model
        for i,(prefix,parameters) in enumerate(list(self.fitComponents.items())):
            if i == 0:
                # Evaluate the model names against lmfit
                comp_model = parameters['model'](prefix=prefix)
            else:
                # Add all contributions to composite model
                comp_model += parameters['model'](prefix=prefix)

        # Make all necessary parameters and initialize
        params = comp_model.make_params()
        
        # Set the parameters according to the passed dictionary if available
        for parmodel,par_dict in self.fitComponents.items():
            # Remove the model key as this is not part of the lmfit required arguments
            par_dict.pop('model')
            for parprops,par_props_dict in par_dict.items():
                # Overwrite initialized pars
                params[f"{parmodel}{parprops}"].set(**par_props_dict)

        # Get the boundaries
        if lower_limit != None:
            idx_low = (np.abs(self.SCADataObject.x_stream-lower_limit)).argmin()
        else:
            idx_low = None

        if upper_limit != None:
            idx_high = (np.abs(self.SCADataObject.x_stream-upper_limit)).argmin()
        else:
            idx_high = None

        # Crop the arrays
        x = self.SCADataObject.x_stream[idx_low:idx_high]
        y = self.SCADataObject.y_stream[idx_low:idx_high]
        
        # Run lmfit minimizer
        out = comp_model.fit(y, params, x=x)
                
        # Prepare data storage
        class added_object:
            def __init__(self):
                pass
        
        data = dict()

        # Store the best fit 
        if fit == 'best':

            # Create dict with objects to be compatible with other loaders
            data[0] = added_object()

            # Store all pertinent information in object
            data[0].x_stream = x
            data[0].y_stream = out.best_fit
            data[0].scan = self.SCADataObject.scan
            data[0].xlabel = self.SCADataObject.xlabel
            data[0].ylabel = f"{self.SCADataObject.ylabel} - Fit"
            index = len(self.data) + 1 
            data[0].legend = f'{index} - S{self.SCADataObject.scan} - LMFit (best fit)'
            data[0].filename = self.SCADataObject.filename
            
        # Else, store the initial fit
        elif fit == 'init':
            init_components = out.eval_components(params=out.init_params)
            for i,(prefix,arr) in enumerate(list(init_components.items())):

                # Create dict with objects to be compatible with other loaders
                data[i] = added_object()

                # Store all pertinent information in object
                data[i].x_stream = x
                data[i].y_stream = arr
                data[i].scan = self.SCADataObject.scan
                data[i].xlabel = self.SCADataObject.xlabel
                data[i].ylabel = f"{self.SCADataObject.ylabel} - Initial {prefix}"
                index = len(self.data) + 1 
                data[i].legend = f'{index} - S{self.SCADataObject.scan} - Initial {prefix}component'
                data[i].filename = self.SCADataObject.filename

        # Else, store the final fit alongside the fitted components
        elif fit == 'components':

            # Append best fit and components
            # Create dict with objects to be compatible with other loaders
            data[0] = added_object()

            # Store all pertinent information in object
            data[0].x_stream = x
            data[0].y_stream = out.best_fit
            data[0].scan = self.SCADataObject.scan
            data[0].xlabel = self.SCADataObject.xlabel
            data[0].ylabel = f"{self.SCADataObject.ylabel} - Fit"
            index = len(self.data) + 1 
            data[0].legend = f'{index} - S{self.SCADataObject.scan} - LMFit (best fit)'
            data[0].filename = self.SCADataObject.filename

            best_components = out.eval_components()
            for i,(prefix,arr) in enumerate(list(best_components.items())):

                # Create dict with objects to be compatible with other loaders
                data[i+1] = added_object()

                # Store all pertinent information in object
                data[i+1].x_stream = x
                data[i+1].y_stream = arr
                data[i+1].scan = self.SCADataObject.scan
                data[i+1].xlabel = self.SCADataObject.xlabel
                data[i+1].ylabel = f"{self.SCADataObject.ylabel} - Fit {prefix}"
                index = len(self.data) + 1 
                data[i+1].legend = f'{index} - S{self.SCADataObject.scan} - LMFit {prefix}component'
                data[i+1].filename = self.SCADataObject.filename

        else:
            raise Exception("Specified fit undefined. Choose <<best>>, <<components>> or <<init>>.")

        self.data.append(data)
        self.out = out

    def fit_report(self):
        """Print the fit report"""
        print(self.out.fit_report())

    def fit_values(self):
        """Return the best fit values as pandas DataFrame"""
        df = pd.DataFrame.from_dict(self.out.best_values,orient='index')
        df.columns = ['Parameters']
        return df