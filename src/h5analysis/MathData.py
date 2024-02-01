# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d
from shapely.geometry import Point, Polygon
import skimage as ski

# Import Loaders
from .LoadData import Load1d, Load2d, LoadHistogram

# Import simplemath and datautil
from .simplemath import grid_data_mesh
from .datautil import mca_roi, get_indices, get_indices_polygon

class Object1dAddSubtract(Load1d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()
        self.DataObjectsSubtract = list()

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

        self.DataObjectsAdd.append(obj.data[line][scan])
        
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
        self.DataObjectsSubtract.append(obj.data[line][scan])
        
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
        data[0].scan = 'Misc'
        data[0].legend = 'Addition/Subtraction'

        data[0].xlabel = 'x-stream'
        data[0].ylabel = 'y-stream'
        data[0].filename = 'Object Math'
        
        self.data.append(data)

#########################################################################################

class Object1dStitch(Load1d):
    """Apply stitching on loader objects"""

    def __init__(self):
        self.DataObjectsStitch = list()

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

        self.DataObjectsStitch.append(obj.data[line][scan])
                
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
        data[0].scan = 'Misc'
        data[0].legend = 'Stitching'

        data[0].xlabel = 'x-stream'
        data[0].ylabel = 'y-stream'
        data[0].filename = 'Object Math'
        
        self.data.append(data)

#########################################################################################


class Object2dAddSubtract(Load2d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()
        self.DataObjectsSubtract = list()

        return Load2d.__init__(self)
        
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
        self.DataObjectsAdd.append(obj.data[line][scan])
        
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
        self.DataObjectsSubtract.append(obj.data[line][scan])
        
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
        data[0].scan = 'Misc'
        data[0].legend = 'Addition/Subtraction'
        data[0].xlabel = 'x-stream'
        data[0].ylabel = 'y-stream'
        data[0].zlabel = 'Detector'
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
        data[0].legend = '2d ROI reduction'
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
        data[0].legend = '2d polygon reduction'
        data[0].filename = self.MCADataObject.filename
        
        self.data.append(data)

class Object2dTransform(Load2d):
    """Apply transformations to a 2d image"""

    def transform(self,trans_y):
        """ Apply math operations on a per data point basis. Change second axis (y) for all data along first (x) axis

            Parameters
            ----------
            trans_y: string
                math expression to be evaluated at every x data point
                available variables include 'x', 'y', 'z'.
        """

        # Do this for all scans in loaded object
        for i, val in enumerate(self.data):
            for k, v in val.items():

                # Get data as variables x, y, z
                y = v.new_y
                z = v.new_z

                # Store the axes modified for each data point
                axes_math = list()
                for x in v.new_x:
                    # For each data point, evaluate the math and store y-axis
                    axes_math.append(eval(trans_y))

                # Find min/max values
                mins = list()
                maxs = list()
                for arr in axes_math:
                    mins.append(min(arr))
                    maxs.append(max(arr))
                
                # Need the biggest common overlap
                # Then create new, common y scale
                ymin = max(mins)
                ymax = min(maxs)
                new_y = np.linspace(ymin, ymax, len(v.new_y), endpoint=True)

                # Store shifted data in new array
                scatter_z = np.zeros((len(new_y),len(v.new_x)))

                # Evaluate the image on the new common energy axis
                for idx,val in enumerate(np.transpose(z)):
                    scatter_z[:,idx] = interp1d(axes_math[idx],val)(new_y)
                
                # Note, loading with Load2d ensures that the x-axis (v.new_x) is
                # is already interpolated on evenly spaced grid

                # Update data as calculated above
                v.new_z = scatter_z
                v.new_y = new_y
                v.ymin = ymin
                v.ymax = ymax
                v.ylabel = 'Transformed Scale'

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
class ObjectHistMath(LoadHistogram):
    """Apply addition on histogram loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()

        return LoadHistogram.__init__(self)
        
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
        self.DataObjectsAdd.append(obj.data[line][scan])
        
    def evaluate(self):
        """Evaluate the request"""

        # Make sure there is no other scan loaded
        if self.data != []:
            raise UserWarning("Can only load one scan at a time.")

        # Iterate over all loaded scans
        x_data = list()
        y_data = list()
        z_data = list()

        # Append all single stream data to respecive list
        for i,item in enumerate(self.DataObjectsAdd):
            x_data.append(item.x_data)
            y_data.append(item.y_data)
            z_data.append(item.z_data)
        
        # Concatenate list data to numpy array
        all_x = np.concatenate(tuple(x_data))
        all_y = np.concatenate(tuple(y_data))
        all_z = np.concatenate(tuple(z_data))
        
        # Grid the data to numpy 2d histogram
        xmin, xmax, ymin, ymax, xedge, yedge, new_z, zmin, zmax = grid_data_mesh(all_x,all_y,all_z)

        # Store data
        class added_object:
            def __init__(self):
                pass

        data = dict()
        data[0] = added_object()
        data[0].xmin = xmin
        data[0].xmax = xmax
        data[0].ymin = ymin
        data[0].ymax = ymax
        data[0].new_x = xedge
        data[0].new_y = yedge
        data[0].new_z = new_z
        data[0].zmin = zmin
        data[0].zmax = zmax
        data[0].x_data = all_x
        data[0].y_data = all_y
        data[0].z_data = all_z
        data[0].scan = 'Misc'
        data[0].legend = 'Addition/Subtraction'
        data[0].xlabel = 'x-stream'
        data[0].ylabel = 'y-stream'
        data[0].zlabel = 'z-stream'
        data[0].filename = 'Simple Math'
        
        self.data.append(data)