# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d

# Import Loaders
from .LoadData import Load1d, Load2d, LoadHistogram

# Import simplemath
from .simplemath import grid_data_mesh

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