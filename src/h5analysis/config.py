"""Tools for the configuration of the data structure"""

# Import parse module
import parse

class h5Config:
    """Internal function to configure data streams."""

    def __init__(self):
        """Initialize variables and data containers"""
        self.h5dict = dict()
        self.sca_folders = list()
        self.roi_dict = dict()
        self.index = 0
        
    def key(self,pattern,scan_var):
        """Set the h5 group structure.

        Parameters
        ----------
        pattern: string
            pattern of the groups holding scan entries in h5
        scan_var: string
            name of the variable in the "pattern" containing the scan number
        """

        self.h5key = pattern
        self.scan_var = scan_var
        
    def get_h5key(self,scan):
        """Returns the proper h5 group/key to a given scan

        Parameters
        ----------
        scan: int,
            scan number
        """

        # Format the string according to the specified pattern
        kwargs = dict()
        kwargs[self.scan_var] = scan
        return self.h5key.format(**kwargs)
    
    def get_h5scan(self,key):
        """Extracts the scan number (integer) from h5 group/key for a scan

        Parameters
        ----------
        key: string,
            full h5 group/key to scan
        """

        return parse.parse(self.h5key,key)[self.scan_var]
    
    def get_path(self,scan,path):
        """Returns the full path to a h5 group/key for a given scan

        Parameters
        ----------
        scan: int,
            scan number
        path: string
            sub group in h5 container
        """

        if self.h5key == '':
            return path
        else:
            return f"{self.get_h5key(scan)}/{path}"
        
    def define_roi(self,roi_string,roi_tuple):
        """Setup dictionary of user-defined ROIs.

        Parameters
        ----------
        roi_string: string
            pre-defined ROI string
        roi_tuple: tuple
            associated values as tuples, 
            i.e. (low_energy,high_energy)
        """

        if isinstance(roi_string,str) and isinstance(roi_tuple,tuple):
            self.roi_dict[roi_string] = roi_tuple
        else:
            raise Exception("Wrong data type.")
        
    def sca_folder(self,path):
        """Adds a specific path as sca folder for which all scalars will be directly accessible

        Parameters
        ----------
        path: string
            sub group in h5 container
        """

        self.sca_folders.append(path)
        
    def sca(self,alias,path,norm_by=None,x_label=None,y_label=None):
        """Adds a specific alias for SCA data

        Parameters
        ----------
        alias: string
            name used to access this data stream
        path: string
            sub group in h5 container, SCA data
        norm_by: string
            sub group in h5 container, SCA data to norm by
        x_label: string
            general custom label for the x-stream
        y_label: string
            general custom label for the x-stream
        """

        # Set up a dict with all relevant data and store in alias db
        dataDict = dict()
        dataDict['type'] = 'SCA'
        dataDict['SCA_Path'] = path
        dataDict['norm_by'] = norm_by
        dataDict['x_label'] = x_label
        dataDict['y_label'] = y_label
        self.h5dict[alias] = dataDict
        
    def mca(self,alias,path,scale,norm_by=None,x_label=None,y_label=None):
        """Adds a specific alias for MCA data

        Parameters
        ----------
        alias: string
            name used to access this data stream
        path: string
            sub group in h5 container, MCA data
        scale: string
            sub group in h5 container, SCA data
        norm_by: string
            sub group in h5 container, SCA data to norm by
        x_label: string
            general custom label for the x-stream
        y_label: string
            general custom label for the y-stream
        """

        # Set up a dict with all relevant data and store in alias db
        dataDict = dict()
        dataDict['type'] = 'MCA'
        dataDict['MCA_Path'] = path
        dataDict['MCA_Scale'] = scale
        dataDict['norm_by'] = norm_by
        dataDict['x_label'] = x_label
        dataDict['y_label'] = y_label
        self.h5dict[alias] = dataDict
        
    def stack(self,alias,path,scale1,scale2,norm_by=None,label1=None,label2=None):
        """Adds a specific alias for STACK data

        Parameters
        ----------
        alias: string
            name used to access this data stream
        path: string
            sub group in h5 container, MCA data
        scale1: string
            sub group in h5 container, SCA data
        scale1: string
            sub group in h5 container, SCA data
        norm_by: string
            sub group in h5 container, SCA data to norm by
        label1: string
            general custom label for the stream
        label2: string
            general custom label for the stream
        """

        # Set up a dict with all relevant data and store in alias db
        dataDict = dict()
        dataDict['type'] = 'STACK'
        dataDict['STACK_Path'] = path
        dataDict['MCA_Scale'] = scale1
        dataDict['STACK_Scale'] = scale2
        dataDict['norm_by'] = norm_by
        dataDict['label1'] = label1
        dataDict['label2'] = label2
        self.h5dict[alias] = dataDict