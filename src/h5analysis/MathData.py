"""Loader classes for the performance of math operations on data - to be used by users."""

# Scientific modules
import numpy as np
import pandas as pd
from .interpolate import interp1d, interp2d
from shapely.geometry import Point, Polygon
import skimage as ski
import lmfit
from lmfit.models import GaussianModel, QuadraticModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel, LorentzianModel, ExponentialModel

# Import Loaders
from .LoadData import Load1d, Load2d, Load3d, LoadHistogram3d

# Import simplemath and datautil
from .simplemath import handle_eval, apply_savgol
from .datautil import mca_roi, get_indices, get_indices_polygon
from .data_1d import apply_kwargs_1d
from .parser import parse

class Object1dAddSubtract(Load1d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()
        self.DataObjectsSubtract = list()
        self.x_string = ""
        self.y_string = ""
        self.scan_string = "S"
        self.xaxis_label = list()
        self.yaxis_label = list()

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
        for x in o.xaxis_label:
            self.xaxis_label.append(x)
        for y in o.yaxis_label:
            self.yaxis_label.append(y)

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
        for x in o.xaxis_label:
            self.xaxis_label.append(x)
        for y in o.yaxis_label:
            self.yaxis_label.append(y)

        self.DataObjectsSubtract.append(o)
        
    def evaluate(self,filename=None,legend_item=None,twin_y=False,matplotlib_props=dict()):
        """ Evaluate the request
        
            Parameters
            ----------
            kwargs:
                filename: str
                    Name of the data file
                legend_item: str
                    Text to appear in legend
                twin_y: Boolean
                    Switch to display data on second y axis
                matplotlib_props: dict
                    Dictionary for matplotlib properties       
        """

        if self.DataObjectsAdd == []:
            raise Exception('You need to add at least one scan.')

        all_objs = self.DataObjectsAdd + self.DataObjectsSubtract
        start_list = list()
        end_list = list()
        x_list = list()

        for i,item in enumerate(all_objs):
            start_list.append(item.x_stream.min())
            end_list.append(item.x_stream.max())
            x_list.append(item.x_stream)

            if i == 0:
                x_diff = np.abs(np.diff(item.x_stream).min())

        #Check if we need to interpolate scales
        if all([np.array_equal(x_list[0],test_scale) for test_scale in x_list]):
            # All scales are equal, no interpolation required
            MASTER_x = x_list[0]

        else:
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
                MASTER_y = interp1d(item.x_stream,item.y_stream,MASTER_x)
            
            # Interpolate other data and add
            else:
                MASTER_y += interp1d(item.x_stream,item.y_stream,MASTER_x)
        
        # Second, subtract objects
        for i,item in enumerate(self.DataObjectsSubtract):
            MASTER_y -= interp1d(item.x_stream,item.y_stream,MASTER_x)

        # Store data
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass
        
        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()
        data[0].x_stream = MASTER_x
        data[0].y_stream = MASTER_y
        data[0].scan = self.scan_string
        data[0].twin_y = twin_y
        
        # Set matplotlib props
        data[0].matplotlib_props = matplotlib_props

        if legend_item == None:
            index = len(self.data) + 1
            data[0].legend = f'{index} - {self.scan_string} - Addition/Subtraction'
        else:
            data[0].legend = legend_item

        data[0].xlabel = self.x_string
        data[0].ylabel = self.y_string
        data[0].xaxis_label = self.xaxis_label
        data[0].yaxis_label = self.yaxis_label

        if filename == None:
            data[0].filename = 'Object Math'
        else:
            data[0].filename = filename
        
        self.data.append(data)

#########################################################################################

class Object1dStitch(Load1d):
    """Apply stitching on loader objects"""

    def __init__(self):
        self.DataObjectsStitch = list()
        self.x_string = ""
        self.y_string = ""
        self.scan_string = "S"
        self.xaxis_label = list()
        self.yaxis_label = list()

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
        for x in o.xaxis_label:
            self.xaxis_label.append(x)
        for y in o.yaxis_label:
            self.yaxis_label.append(y)

        self.DataObjectsStitch.append(o)
                
    def evaluate(self,average=True,adjust_scale=False,filename=None,legend_item=None,twin_y=False,matplotlib_props=dict()):
        """ Evaluate the request
        
            Parameters
            ----------
            kwargs:
                average: Boolean
                    For overlap, whether the first scan takes precedence (False) or
                    if overlap is averaged (True)
                adjust_scale: Boolean
                    Adjusts the intensity of consecutive scans to match the precessors intensity in the overlap
                    Automatically sets average True
                filename: str
                    Name of the data file
                legend_item: str
                    Text to appear in legend
                twin_y: Boolean
                    Switch to display data on second y axis
                matplotlib_props: dict
                    Dictionary for matplotlib properties    
        
        """

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

        if average == False and adjust_scale == True:
            average = True

        if average == True:
            if adjust_scale == True:
                for i, item in enumerate(self.DataObjectsStitch):
                    if i == 0:
                        MASTER_y = interp1d(item.x_stream,item.y_stream,MASTER_x)
                        twos = np.ones_like(MASTER_y)
                        twos[np.isnan(MASTER_y)] = 0
                        divisor = twos
                    else:
                        item = interp1d(item.x_stream,item.y_stream,MASTER_x)
                        ones = np.ones_like(item)
                        ones[np.isnan(item)] = 0
                        mask = np.add(ones,twos)==2
                        if mask.any()==False:
                            factor = 1
                        else:
                            factor = np.true_divide(MASTER_y,divisor,where=divisor!=0)[mask].mean()/item[mask].mean()
                        MASTER_y = np.nansum(np.dstack((MASTER_y,factor*item)),2)[0]
                        divisor = np.add(divisor,ones)
                        twos = np.ones_like(MASTER_y)
                        twos[divisor==0] = 0

                # Divide big matrix by divisor to get average
                MASTER_y = np.true_divide(MASTER_y,divisor,where=divisor!=0)

            else:
                # Iterate over all loaded scans for interpolation
                for i, item in enumerate(self.DataObjectsStitch):                
                    # interpolate to common scale
                    item = interp1d(item.x_stream,item.y_stream,MASTER_x)
                    # Store results
                    MASTER_y_list.append(item)
                    # Return boolean True where array element is a number
                    MASTER_y_nan_list.append(~np.isnan(item))

                # This is for averaging
                # Sum the arrays of common length, treat nan as 0
                # For each element, sum how many True (numbers) contribute to the sum
                # Normalize to get average by array division
                MASTER_y = np.nansum(MASTER_y_list,axis=0)/np.sum(MASTER_y_nan_list,axis=0)

        else:
            for i, item in enumerate(self.DataObjectsStitch):
                if i == 0:
                    item = interp1d(item.x_stream,item.y_stream,MASTER_x)
                    MASTER_y_list.append(item)
                    mask = np.ones_like(item)
                    mask[np.isnan(item)] = 0
                else:
                    item = interp1d(item.x_stream,item.y_stream,MASTER_x)
                    item[mask!=0] = 0
                    MASTER_y_list.append(item)
                    mask2 = np.ones_like(item)
                    mask2[np.isnan(item)] = 0
                    mask = np.add(mask,mask2)

            MASTER_y = np.nansum(MASTER_y_list,axis=0)


        # Store data
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass

        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()
        data[0].x_stream = MASTER_x
        data[0].y_stream = MASTER_y
        data[0].scan = self.scan_string
        data[0].twin_y = twin_y
        
        # Set matplotlib props
        data[0].matplotlib_props = matplotlib_props

        if legend_item == None:
            index = len(self.data) + 1
            data[0].legend = f'{index} - {self.scan_string} - Stitching'
        else:
            data[0].legend = legend_item

        data[0].xlabel = self.x_string
        data[0].ylabel = self.y_string
        data[0].xaxis_label = self.xaxis_label
        data[0].yaxis_label = self.yaxis_label

        if filename == None:
            data[0].filename = 'Object Math'
        else:
            data[0].filename = filename
        
        self.data.append(data)

#########################################################################################

class Object1dTransform(Load1d):
    """Apply baseline to 1d data"""

    def load(self,obj,line,scan):
        """Loader for 1d object
        
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

    def add(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def subtract(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def stitch(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def background(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def loadObj(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def baseline(self,algorithm,smooth=None,subtract=True,**kwargs):
        """Add baseline from the pybaselines module
        
            Parameters
            ----------
            algorithm: str
                name of the algorithm
            smooth: tuple
                Sets Savitsky-Golay filter properties: (window length, polynomial order)
            subtract: Boolean
                Whether the baseline is subtracted from the data or added as existing data stream
            kwargs: dict
                Key-word arguments for tuning of baseline algorithm
        """

        from pybaselines import Baseline

        x = self.SCADataObject.x_stream
        y = self.SCADataObject.y_stream

        if smooth != None:
            x,y = apply_savgol(x,y,smooth[0],smooth[1],0)

        base = Baseline(x,check_finite=False)
        method = getattr(base,algorithm)

        new_baseline = method(y,**kwargs)[0]

        # Prepare data storage
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass
        
        data = dict()
        # Create dict with objects to be compatible with other loaders
        data[0] = added_object()

        # Store all pertinent information in object
        data[0].x_stream = x
        data[0].scan = self.SCADataObject.scan
        data[0].xlabel = self.SCADataObject.xlabel
        data[0].ylabel = f"{self.SCADataObject.ylabel}"
        index = len(self.data) + 1 
        data[0].filename = self.SCADataObject.filename

        if subtract == True:
            data[0].legend = f'{index} - S{self.SCADataObject.scan} - Baseline (subtracted)'
            data[0].y_stream = np.subtract(y,new_baseline)
        else:
            data[0].legend = f'{index} - S{self.SCADataObject.scan} - Baseline'
            data[0].y_stream = new_baseline

            

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
        """This method is not defined"""
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
        
    def evaluate(self,filename=None,label_x=None,label_y=None,label_z=None,legend=None):
        """ Evaluate the request
        
            Parameters
            ----------
            kwargs:
                filename: str
                    Name of the data file
                label_x: str
                    Label on horizontal axis
                label_y: str
                    Label on vertical axis
                label_z: str
                    Label for count axis
                legend: str
                    Text for legend/title
        """

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
                    MASTER_detector = interp2d(item.new_x,item.new_y,item.new_z,MASTER_x_stream,MASTER_y_stream)
                    
            # Add all objects (2d) after interpolation step to master
                else:
                    new_z = interp2d(item.new_x,item.new_y,item.new_z,MASTER_x_stream,MASTER_y_stream)

                    MASTER_detector = np.add(MASTER_detector,new_z)
            
            # Add all objects (2d) that need to be removed after interpolation step to master
            for i,item in enumerate(self.DataObjectsSubtract):
                    new_z = interp2d(item.new_x,item.new_y,item.new_z,MASTER_x_stream,MASTER_y_stream)

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
                """Initialize data container"""
                pass
        
        # Get props
        if filename==None:
            filename = 'Simple Math'
        if label_x==None:
            label_x = self.x_string
        if label_y==None:
            label_y = self.y_string
        if label_z==None:
            label_z = self.z_string
        if legend==None:
            index = len(self.data) + 1
            f'{index} - {self.scan_string} - Addition/Subtraction'

        data = dict()
        data[0] = added_object()
        data[0].new_x = MASTER_x_stream
        data[0].new_y = MASTER_y_stream
        data[0].new_z = MASTER_detector
        data[0].scan = self.scan_string
        data[0].legend = legend
        data[0].xlabel = label_x
        data[0].ylabel = label_y
        data[0].zlabel = label_z
        data[0].filename = filename
        
        self.data.append(data)

#########################################################################################


class Object2dStitch(Load2d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjects = list()
        self.x_string = ""
        self.y_string = ""
        self.z_string = ""
        self.scan_string = "S"

        return Load2d.__init__(self)
    
    def load(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
        
    def add(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def subtract(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
        
    def stitch(self,obj,line,scan):
        """Loader objects to be stitched
        
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

        self.DataObjects.append(o)
        
    def evaluate(self,average=False,adjust_scale=False,filename=None,label_x=None,label_y=None,label_z=None,legend=None):
        """Evaluate the request
        
        Parameters
        ----------
        average: Boolean
            For overlap, whether the first image takes precedence (False) or
            if overlap is averaged (True)
        adjust_scale: Boolean
            Adjusts the intensity of consecutive images to match the precessors intensity in the overlap
            Automatically sets average True
        filename: str
            Name of the data file
        label_x: str
            Label on horizontal axis
        label_y: str
            Label on vertical axis
        label_z: str
            Label for count axis
        legend: str
            Text for legend/title
        """

        # Make sure there is no other scan loaded
        if self.data != []:
            raise UserWarning("Can only load one scan at a time.")
        
        if self.DataObjects == []:
            raise Exception('You need to add at least one scan.')
        
        # Define generic object in which all data will be stored
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass

        # Start by getting dimensions
        min_x = list()
        max_x = list()
        min_y = list()
        max_y = list()
        diff_x = list()
        diff_y = list()

        # All scans in loaded object
        for i, v in enumerate(self.DataObjects):
            min_x.append(min(v.new_x))
            max_x.append(max(v.new_x))
            min_y.append(min(v.new_y))
            max_y.append(max(v.new_y))
            diff_x.append(np.abs(np.diff(v.new_x)).min())
            diff_y.append(np.abs(np.diff(v.new_y)).min())

        # Determine corners
        lower_x = min(min_x)
        upper_x = max(max_x)
        lower_y = min(min_y)
        upper_y = max(max_y)
        min_diff_x = min(diff_x)
        min_diff_y = min(diff_y)

        # Determine number of points from differences
        numPoints_x = int(np.ceil((upper_x-lower_x)/min_diff_x))
        numPoints_y = int(np.ceil((upper_y-lower_y)/min_diff_y))

        # Limit array size to 100MB (=104857600 bytes)
        # Numpy float64 array element requires 8 bytes
        max_steps = 104857600/8

        if numPoints_x*numPoints_y>max_steps:
            step_norm = int(np.ceil(np.sqrt(numPoints_x*numPoints_y/13107200)))
            x_num = int(numPoints_x/step_norm)
            y_num = int(numPoints_y/step_norm)
        else:
            x_num = numPoints_x
            y_num = numPoints_y

        # Generate new scales
        new_x = np.linspace(lower_x,upper_x,x_num)
        new_y = np.linspace(lower_y,upper_y,y_num)

        if average == False and adjust_scale == True:
            average = True

        if average == False:
            for i, v in enumerate(self.DataObjects):
                if i ==0:
                    # Interpolate image on new big image
                    matrix = interp2d(v.new_x,v.new_y,v.new_z,new_x,new_y,fill_value=np.nan)
                else:
                    # Do this for all images
                    # Check if no information (NaN) has been added to the composite image, if so, add - else, set addition to 0
                    matrix2 = interp2d(v.new_x,v.new_y,v.new_z,new_x,new_y,fill_value=np.nan)
                    matrix_nan = np.isnan(matrix)
                    matrix2_nan = np.isnan(matrix2)
                    m_keep_matrix2 = matrix_nan & ~matrix2_nan
                    matrix[m_keep_matrix2] = matrix2[m_keep_matrix2]

        else:
            if adjust_scale == True:
                for i, v in enumerate(self.DataObjects):
                    if i ==0:
                        # Interpolate image on new big image
                        # Initilize divisor to track how many contributions per data points
                        # Set the contribution to 1 where added, else 0 - use array masking
                        # Add the new contributions to divisor
                        matrix = interp2d(v.new_x,v.new_y,v.new_z,new_x,new_y,fill_value=np.nan)
                        divisor = np.zeros_like(matrix)
                        ones = np.ones_like(matrix)
                        ones[np.isnan(matrix)] = 0
                        divisor = np.add(divisor,ones)
                        twos = divisor # We need this to track where there are entries in the matrix
                    else:
                        # Same as above
                        matrix2 = interp2d(v.new_x,v.new_y,v.new_z,new_x,new_y,fill_value=np.nan)
                        ones = np.ones_like(matrix)
                        ones[np.isnan(matrix2)] = 0
                        overlap_mask = np.add(twos,ones)==2 ## Check where we have overlap, i.e. both matrix and matrix2 have entries
                        if overlap_mask.any()==False:
                            factor = 1
                        else:
                            factor = np.true_divide(matrix,divisor,where=divisor!=0)[overlap_mask].mean()/matrix2[overlap_mask].mean() # calculate scaling factor taking divisor into account
                        matrix = np.nansum(np.dstack((matrix,factor*matrix2)),2) # scale new image to old overlap
                        divisor = np.add(divisor,ones) # now can calculate new divisor corresponding to matrix calculated in the previous line, need to do this after calculating factor
                        twos = np.ones_like(matrix) # re-create matrix for overlap estimation
                        twos[divisor==0] = 0 # only have entries where in matrix where divisor is non-zero

                # Divide big matrix by divisor to get average
                matrix = np.true_divide(matrix,divisor,where=divisor!=0)

            else:
                for i, v in enumerate(self.DataObjects):
                    if i ==0:
                        # Interpolate image on new big image
                        # Initilize divisor to track how many contributions per data points
                        # Set the contribution to 1 where added, else 0 - use array masking
                        # Add the new contributions to divisor
                        matrix = interp2d(v.new_x,v.new_y,v.new_z,new_x,new_y,fill_value=np.nan)
                        divisor = np.zeros_like(matrix)
                        ones = np.ones_like(matrix)
                        ones[np.isnan(matrix)] = 0
                        divisor = np.add(divisor,ones)
                    else:
                        # Same as above
                        matrix2 = interp2d(v.new_x,v.new_y,v.new_z,new_x,new_y,fill_value=np.nan)
                        matrix = np.nansum(np.dstack((matrix,matrix2)),2)
                        ones = np.ones_like(matrix)
                        ones[np.isnan(matrix2)] = 0
                        divisor = np.add(divisor,ones)

                # Divide big matrix by divisor to get average
                matrix = np.true_divide(matrix,divisor,where=divisor!=0)

        # Remove NaN values and set to 0
        matrix = np.nan_to_num(matrix,nan=0,posinf=0,neginf=0)

        # Get props
        if filename==None:
            filename = 'Simple Math'
        if label_x==None:
            label_x = self.x_string
        if label_y==None:
            label_y = self.y_string
        if label_z==None:
            label_z = self.z_string
        if legend==None:
            index = len(self.data) + 1
            f'{index} - {self.scan_string} - Addition/Subtraction'

        # Place data in a dictionary with the same structure as a regular Load1d call, so that we can plot it
        data = dict()
        data[0] = added_object()
        data[0].new_x = new_x
        data[0].new_y = new_y
        data[0].new_z = matrix
        data[0].xlabel = label_x
        data[0].ylabel = label_y
        data[0].zlabel = label_z
        data[0].filename = filename
        data[0].scan = self.scan_string
        data[0].legend = legend
        
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
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def subtract(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def stitch(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def background(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def loadObj(self):
        """This method is not defined"""
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
                """Initialize data container"""
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
            data[0].xaxis_label = [self.MCADataObject.ylabel]

        elif integration_axis == 'y':
            idx_low, idx_high = get_indices(roi,y)
            sum = mca_roi(z,idx_low,idx_high,0)
            MASTER_x = x
            data[0].xlabel = self.MCADataObject.xlabel
            data[0].xaxis_label = [self.MCADataObject.xlabel]

        else:
            raise Exception('Specified integration axis not defined.')

        # Store all pertinent information in object
        data[0].x_stream = MASTER_x
        data[0].y_stream = sum
        data[0].scan = self.MCADataObject.scan
        data[0].ylabel = f"{self.MCADataObject.zlabel} - ROI"
        data[0].yaxis_label = [f"{self.MCADataObject.zlabel} - ROI"]
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
                """Initialize data container"""
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
            data[0].xaxis_label = [self.MCADataObject.ylabel]
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
            data[0].xaxis_label = [self.MCADataObject.xlabel]                
            MASTER_x = x

        else:
            raise Exception('Specified integration axis not defined.')

        # Store all pertinent information in object
        data[0].x_stream = MASTER_x
        data[0].y_stream = sum
        data[0].scan = self.MCADataObject.scan
        data[0].ylabel = f"{self.MCADataObject.zlabel} - ROI"
        data[0].yaxis_label = [f"{self.MCADataObject.zlabel} - ROI"]
        index = len(self.data) + 1 
        data[0].legend = f'{index} - S{self.MCADataObject.scan} - 2d polygon reduction'
        data[0].filename = self.MCADataObject.filename
        
        self.data.append(data)


    def apply_kwargs(self,norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None):
        """ Apply math to 1d reduced objects

            Parameters
            ----------    
        
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

                v.x_stream,v.y_stream = apply_kwargs_1d(v.x_stream,v.y_stream,norm,xoffset,xcoffset,yoffset,ycoffset,grid_x,savgol,binsize)
                    
                self.data[i][k].x_stream = v.x_stream
                self.data[i][k].y_stream = v.y_stream

#########################################################################################

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

                    # Also crop down matrix accordingly
                    z = z[:,xindices]

                # Create new, common y scale
                new_y = np.linspace(ymin, ymax, len(v.new_y), endpoint=True)

                # Store shifted data in new array
                scatter_z = np.zeros((len(new_y),len(v.new_x)))

                # Evaluate the image on the new common energy axis
                for idx,val in enumerate(np.transpose(z)):
                    scatter_z[:,idx] = interp1d(yaxes[idx],val,new_y)
                
                # Note, loading with Load2d ensures that the x-axis (v.new_x) is
                # is already interpolated on evenly spaced grid

                # Update data as calculated above
                v.new_z = scatter_z
                v.new_y = new_y
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
                xlabel = v.ylabel
                ylabel = v.xlabel

                v.new_x = new_x
                v.new_y = new_y
                v.xlabel = xlabel
                v.ylabel = ylabel

                # Write back to dict
                self.data[i][k] = v

    def baseline(self,algorithm,smooth=None,subtract=True,**kwargs):
        """Add baseline from the pybaselines module
        
            Parameters
            ----------
            algorithm: str
                name of the algorithm
            smooth: tuple
                Sets Savitsky-Golay filter properties: (window length, polynomial order)
            subtract: Boolean
                Whether the baseline is subtracted from the data or replaces the existing data stream
            kwargs: dict
                Key-word arguments for tuning of baseline algorithm
        """

        from pybaselines import Baseline

        # For all loaded objects
        for i, val in enumerate(self.data):
            for k, v in val.items():

                baseline_spec = np.zeros_like(v.new_z)

                for idx,spec in enumerate(np.transpose(v.new_z)):
                    scale = v.new_y

                    if smooth != None:
                        scale,spec = apply_savgol(scale,spec,smooth[0],smooth[1],0)

                    base = Baseline(scale,check_finite=True)
                    method = getattr(base,algorithm)

                    baseline_spec[:,idx] = np.nan_to_num(method(spec,**kwargs)[0])

                v.new_y = scale
                if subtract == True:
                    v.new_z = np.subtract(v.new_z,baseline_spec)
                    v.ylabel = f'Baseline subtracted {v.ylabel}'
                else:
                    v.new_z = baseline_spec
                    v.ylabel = f'Baseline {v.ylabel}'

                # Prepare data storage
                self.data[i][k] = v


#########################################################################################
class Object3dAddSubtract(Load3d):
    """Apply addition/subtraction on loader objects"""

    def __init__(self):
        self.DataObjectsAdd = list()
        self.DataObjectsSubtract = list()
        self.x_string = ""
        self.y_string = ""
        self.z_string = ""
        self.scan_string = "S"

        return Load3d.__init__(self)
    
    def load(self):
        """This method is not defined"""
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
        
    def evaluate(self,filename=None,label_x=None,label_y=None,label_z=None):
        """ Evaluate the request
        
            Parameters
            ----------
            kwargs:
                filename: str
                    Name of the data file
                label_x: str
                    Label on horizontal axis
                label_y: str
                    Label on vertical axis
                label_z: str
                    Label for count axis
        """

        # Make sure there is no other scan loaded
        if self.data != []:
            raise UserWarning("Can only load one scan at a time.")
        
        if self.DataObjectsAdd == []:
            raise Exception('You need to add at least one scan.')
        
        all_objs = self.DataObjectsAdd + self.DataObjectsSubtract

        # Check that dimensions are matching
        stack_data_dims = list()

        for obj in all_objs:
            stack_data_dims.append(np.shape(obj.stack))

        # make sure the dimensions are matching
        if all([np.array_equal(stack_data_dims[0],dims) for dims in stack_data_dims]):
            pass
        else:
            raise Exception("Adding stacks with incompatible dimensions.")

        for i,item in enumerate(self.DataObjectsAdd):
            if i ==0:
                MASTER = item.stack
                
            else:
                MASTER = np.add(MASTER,item.stack)
        
        # Add all objects (2d) that need to be removed after interpolation step to master
        for i,item in enumerate(self.DataObjectsSubtract):
                if i == 0:
                    MASTER_SUB = item.stack
                else:
                    MASTER_SUB = np.add(MASTER_SUB,item.stack)

        # Remove subtraction from Master, if any
        if len(self.DataObjectsSubtract)>0:
            MASTER = np.subtract(MASTER,MASTER_SUB)

        
        # Store data
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass
        
        # Get props
        if filename==None:
            filename = 'Simple Math'
        if label_x==None:
            label_x = self.x_string
        if label_y==None:
            label_y = self.y_string
        if label_z==None:
            label_z = self.z_string

        data = dict()
        data[0] = added_object()

        data[0].new_x = self.DataObjectsAdd[0].new_x
        data[0].new_y = self.DataObjectsAdd[0].new_y
        data[0].stack = MASTER

        data[0].ind_stream = self.DataObjectsAdd[0].ind_stream
        data[0].str_ind_stream = self.DataObjectsAdd[0].str_ind_stream

        data[0].scan = self.scan_string
        data[0].xlabel = label_x
        data[0].ylabel = label_y
        data[0].zlabel = label_z
        data[0].filename = filename
        
        self.data.append(data)

#########################################################################################

class Object3dHistogramTransform(LoadHistogram3d):
    """Apply transformations to a 3d image stack"""

    def baseline(self,algorithm,smooth=None,subtract=True,**kwargs):
        """Add baseline from the pybaselines module
        
            Parameters
            ----------
            algorithm: str
                name of the algorithm
            smooth: tuple
                Sets Savitsky-Golay filter properties: (window length, polynomial order)
            subtract: Boolean
                Whether the baseline is subtracted from the data or replaces the existing data stream
            kwargs: dict
                Key-word arguments for tuning of baseline algorithm
        """

        from pybaselines import Baseline

        # For all loaded objects
        for i, val in enumerate(self.data):
            for k, v in val.items():

                baseline_spec = np.zeros_like(v.stack)
                for idx1 in range(0,np.shape(v.stack)[1]):
                    for idx2 in range(0,np.shape(v.stack)[2]):
                        scale = v.ind_stream
                        spec = v.stack[:,idx1,idx2]

                        if smooth != None:
                            scale,spec = apply_savgol(scale,spec,smooth[0],smooth[1],0)

                        base = Baseline(scale,check_finite=True)
                        method = getattr(base,algorithm)

                        baseline_spec[:,idx1,idx2] = np.nan_to_num(method(spec,**kwargs)[0])

                if subtract == True:
                    v.stack = np.subtract(v.stack,baseline_spec)
                    v.zlabel = f'Baseline subtracted {v.zlabel}'
                else:
                    v.stack = baseline_spec
                    v.zlabel = f'Baseline {v.zlabel}'

                # Prepare data storage
                self.data[i][k] = v

#########################################################################################

class Object3dHistogramReduce(Load1d):
    """Apply transformations to a 3d image stack"""

    def load(self,obj,line,scan):
        """Loader for 3d object
        
            Parameters
            ----------
            obj: object
                Loader object
            line: int
                load, add, subtract line of object (indexing with 0)
            scan: int
                number of the scan to be accessed
        """
        self.STACKDataObject = obj.data[line][scan]

    def add(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def subtract(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def stitch(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def background(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def loadObj(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def polygon(self,polygon,exact=False):
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

        SPEC = list()
        x_stream = self.STACKDataObject.ind_stream

        for idx,slice2d in enumerate(self.STACKDataObject.stack):
            scale1 = np.array(self.STACKDataObject.new_x)[idx]
            scale2 = np.array(self.STACKDataObject.new_y)[idx]


            if exact == False:
                # If not exact, 
                # return array indices corresponding to polygon boundaries
                # then apply algorithm to get mask
                # apply mask to array
                idx = get_indices_polygon(polygon,scale1,scale2)
                mask = ski.draw.polygon2mask(slice2d.shape, idx)
                sum = np.where(mask,np.array(slice2d),0).sum(axis=(0,1))

            else:
                # If exact,
                # get shapely polygon
                # check iteratively for each data point in array if contained
                # in polygon.
                # if so, add to sum
                poly = Polygon(polygon)
                sum = 0
                for x_idx,px in enumerate(scale1):
                    for y_idx,py in enumerate(scale2):
                        p = Point(px,py)
                        if poly.contains(p):
                            sum += np.array(slice2d)[y_idx,x_idx]

            SPEC.append(sum)

    
        # Prepare data storage
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass
        
        # Create dict with objects to be compatible with other loaders
        data = dict()
        data[0] = added_object()

        # Store all pertinent information in object
        data[0].x_stream = x_stream
        data[0].y_stream = np.array(SPEC)

        data[0].scan = self.STACKDataObject.scan

        # Set labels and independent data stream
        data[0].xlabel = f"{self.STACKDataObject.zlabel} - Scale"
        data[0].xaxis_label = [f"{self.STACKDataObject.zlabel} - Scale"]
        data[0].ylabel = f"Intensity"
        data[0].yaxis_label = [f"Intensity"]

        index = len(self.data) + 1 
        data[0].legend = f'{index} - S{self.STACKDataObject.scan} - 3d polygon reduction'
        data[0].filename = self.STACKDataObject.filename
        
        self.data.append(data)


    def apply_kwargs(self,norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, grid_x=[None, None, None], savgol=None, binsize=None):
        """ Apply math to 1d reduced objects

            Parameters
            ----------    
        
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

                v.x_stream,v.y_stream = apply_kwargs_1d(v.x_stream,v.y_stream,norm,xoffset,xcoffset,yoffset,ycoffset,grid_x,savgol,binsize)
                    
                self.data[i][k].x_stream = v.x_stream
                self.data[i][k].y_stream = v.y_stream


#########################################################################################
#########################################################################################
                
class Object1dFit(Load1d):
    """Apply fit to 1d data"""

    def load(self,obj,line,scan):
        """Loader for 1d object
        
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
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def subtract(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def stitch(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def background(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def loadObj(self):
        """This method is not defined"""
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
        x_stream = self.SCADataObject.x_stream
        x = self.SCADataObject.x_stream[idx_low:idx_high]
        y = self.SCADataObject.y_stream[idx_low:idx_high]
        
        # Run lmfit minimizer
        out = comp_model.fit(y, params, x=x)
                
        # Prepare data storage
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass
        
        data = dict()

        # Store the best fit 
        if fit == 'best':

            # Create dict with objects to be compatible with other loaders
            data[0] = added_object()

            # Store all pertinent information in object
            data[0].x_stream = x_stream
            data[0].y_stream = out.eval(x=x_stream)
            data[0].scan = self.SCADataObject.scan
            data[0].xlabel = self.SCADataObject.xlabel
            data[0].ylabel = f"{self.SCADataObject.ylabel} - Fit"
            index = len(self.data) + 1 
            data[0].legend = f'{index} - S{self.SCADataObject.scan} - LMFit (best fit)'
            data[0].filename = self.SCADataObject.filename
            
        # Else, store the initial fit
        elif fit == 'init':
            init_components = out.eval_components(params=out.init_params,x=x_stream)
            for i,(prefix,arr) in enumerate(list(init_components.items())):

                # Create dict with objects to be compatible with other loaders
                data[i] = added_object()

                # Store all pertinent information in object
                data[i].x_stream = x_stream
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
            data[0].x_stream = x_stream
            data[0].y_stream = out.eval(x=x_stream)
            data[0].scan = self.SCADataObject.scan
            data[0].xlabel = self.SCADataObject.xlabel
            data[0].ylabel = f"{self.SCADataObject.ylabel} - Fit"
            index = len(self.data) + 1 
            data[0].legend = f'{index} - S{self.SCADataObject.scan} - LMFit (best fit)'
            data[0].filename = self.SCADataObject.filename

            best_components = out.eval_components(x=x_stream)
            for i,(prefix,arr) in enumerate(list(best_components.items())):

                # Create dict with objects to be compatible with other loaders
                data[i+1] = added_object()

                # Store all pertinent information in object
                data[i+1].x_stream = x_stream
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
    
#########################################################################################

class Object1dEXAFS(Load1d):
    """EXAFS processing provided by xraylarch integration"""

    def load(self,obj,line,scan):
        """Loader for 1d object
        
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
        self.scan_string = scan

        from larch import Group
        x = self.SCADataObject.x_stream
        y = self.SCADataObject.y_stream

        x_monotonic = np.array(x)[np.where(np.diff(x)>0)[0]]
        y_monotonic = np.array(y)[np.where(np.diff(x)>0)[0]]

        self.EXAFSDataGroup = Group(name='my',energy=x_monotonic,mu=y_monotonic)
        

    def add(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def subtract(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def stitch(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def background(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
    
    def loadObj(self):
        """This method is not defined"""
        raise Exception("This method is not defined")
        
    def calculate_autobk(self,**kwargs):
        """Calculate the background
        
            Parameters
            ----------
            **kwargs:
                See autobk function from xraylarch package
        """

        from larch.xafs.autobk import autobk

        autobk(energy=self.EXAFSDataGroup.energy,mu=self.EXAFSDataGroup.mu,group=self.EXAFSDataGroup,**kwargs)

    def calculate_xftf(self,**kwargs):
        """Calculate the forward Fourier Transform
        
            Parameters
            ----------
            **kwargs:
                See xftf function from xraylarch package
        """

        from larch.xafs import xftf
        xftf(self.EXAFSDataGroup,**kwargs)

    def evaluate(self,x,y):
        """Evaluate the given expressions
        
            Parameters
            ----------
            x: string
                See xraylarch group for possible attributes; may apply math operations
            y: string
                See xraylarch group for possible attributes; may apply math operations
        """

        # Apply parser to get split up math operations
        # Place all parsed quatities in dictionary and retrieve corresponding data
        x_parse = parse(x)
        y_parse = parse(y)
        x_dict = dict()
        y_dict = dict()
        for item in x_parse:
            x_dict[item] = self.EXAFSDataGroup[item]
        for item in y_parse:
            y_dict[item] = self.EXAFSDataGroup[item]

        # Store data
        class added_object:
            def __init__(self):
                """Initialize data container"""
                pass
        
        # Create dict with objects to be compatible with other loaders
        # Evaluate expressions for x_stream and y_stream using created dicts of variables only
        data = dict()
        data[0] = added_object()
        data[0].x_stream = np.nan_to_num(eval(x,x_dict),nan=0,posinf=0,neginf=0)
        data[0].y_stream = np.nan_to_num(eval(y,y_dict),nan=0,posinf=0,neginf=0)
        data[0].scan = self.scan_string
        index = len(self.data) + 1
        data[0].legend = f'{index} - {self.scan_string} - EXAFS {y}'

        data[0].xlabel = x
        data[0].ylabel = y
        data[0].xaxis_label = [x]
        data[0].yaxis_label = ['Intensity']
        data[0].filename = 'Object Math'
        
        self.data.append(data)

