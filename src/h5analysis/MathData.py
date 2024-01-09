# Scientific modules
import numpy as np
from scipy.interpolate import interp1d, interp2d

# Import Loaders
from .LoadData import Load1d, Load2d, LoadHistogram

class Object1dMath(Load1d):
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

        # First, add all objects
        for i,item in enumerate(self.DataObjectsAdd):
            # Determine MASTER streams
            if i ==0:
                MASTER_x = item.x_stream
                MASTER_y = item.y_stream
            
            # Interpolate other data and add
            else:
                MASTER_y += interp1d(item.x_stream,item.y_stream,fill_value='extrapolate')(MASTER_x)
        
        # Second, subtract objects
        for i,item in enumerate(self.DataObjectsSubtract):
                MASTER_y -= interp1d(item.x_stream,item.y_stream,fill_value='extrapolate')(MASTER_x)

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
        
class Object2dMath(Load2d):
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

        # Set up the new master streams
        for i,item in enumerate(self.DataObjectsAdd):
            if i ==0:
                MASTER_x_stream = item.new_x
                MASTER_y_stream = item.new_y
                MASTER_detector = item.new_z
                MASTER_xmin = item.xmin
                MASTER_xmax = item.xmax
                MASTER_ymin = item.ymin
                MASTER_ymax = item.ymax
                
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
        data[0].xmin = MASTER_xmin
        data[0].xmax = MASTER_xmax
        data[0].ymin = MASTER_ymin
        data[0].ymax = MASTER_ymax
        data[0].scan = 'Misc'
        data[0].legend = 'Addition/Subtraction'

        # Setup names
        self.x_stream.append('x-stream')
        self.y_stream.append('y-stream')
        self.detector.append('Detector')
        self.filename.append('Simple Math')
        
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
        data[0].xedge = xedge
        data[0].yedge = yedge
        data[0].new_z = new_z
        data[0].zmin = zmin
        data[0].zmax = zmax
        data[0].x_data = all_x
        data[0].y_data = all_y
        data[0].z_data = all_z
        data[0].scan = 'Misc'
        data[0].legend = 'Addition/Subtraction'

        # Set up names
        self.x_stream.append('x-stream')
        self.y_stream.append('y-stream')
        self.detector.append('Detector')
        self.filename.append('Simple Math')
        
        self.data.append(data)