# Scientific modules
import pandas as pd
import numpy as np

# h5 library
import h5py

# Read/Data utilities
from .util import check_key_in_dict
from .readutil import detector_norm, stack_norm

class Data:
    """Standard data class to store h5 files"""

    def __init__(self,config,file,scan):
        """Set up the data object
        
            Parameters
            ----------
            config: dict
                h5 data config 
            file: string
                file name
            scan: int
                scan number
        """

        self.config = config
        self.file = file
        self.scan = scan
        
        # Check if any SCA folder paths are specified
        # Makes all 1d data in specified folder(s) accessible
        if self.config.sca_folders != list():

            # Create sca_data pd.DataFrame
            self.sca_data = pd.DataFrame()
            sca_series = list()
            sca_headers = list()

            with h5py.File(file, 'r') as f:
                # Populate a pandas dataframe with all SCA data
                try:
                    for path in self.config.sca_folders:
                        # Get the full h5 group path
                        p = config.get_path(scan,path)

                        # Iterate over all entries in group
                        for entry in f[p]:
                            # Only regard 1d data
                            if len(f[f'{p}/{entry}'].shape) == 1:
                                # Append to pandas series
                                sca_series.append(pd.Series(np.array(f[f'{p}/{entry}'])))
                                sca_headers.append(str(entry))
                                
                    # Convert pandas series to dataframe and assign header names as per the key in h5
                    self.sca_data = pd.DataFrame(sca_series).transpose(copy=True)
                    self.sca_data.columns = sca_headers
                                
                except Exception as e:
                    raise Exception(e)
                    
    def Scan(self,requisition):
        """ Load data affiliated with specific requisition

            Parameters
            ----------
            requisition: list, string
                either h5 keys or alias defined in config

            Returns
            -------
            req_data: dict
                all data associated with requsition, sorted by req key
        """

        # Use req_data to store all requested data
        req_data = dict()
        
        # Ensure requisitions are of list type
        if not isinstance(requisition,list):
            requisition = [requisition]
        
        # Iterate over all requests
        for req in requisition:
            # If an alias is defined, use config file
            # Check whether we can find an alias
            if check_key_in_dict(req,self.config.h5dict):

                # open h5 file
                with h5py.File(self.file, 'r') as f:
        
                    # Get all attributes for specified req
                    req_attr = self.config.h5dict[req]
            
                    # This is for SCA data
                    if req_attr['type'] == 'SCA':
                        # Get the path and load data from file
                        p = self.config.get_path(self.scan,req_attr['SCA_Path'])
                        data = np.array(f[p])

                        # If normalization is requested, proceed with pulling additional data to norm by
                        # else, skip
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            req_data[req] = np.true_divide(data,norm_by)
                        else:
                            req_data[req] = data
                    
                    # This is for 2d MCA data
                    elif req_attr['type'] == 'MCA':
                        p = self.config.get_path(self.scan,req_attr['MCA_Path'])
                        data = np.array(f[p])
                        
                        # If specified, get the associated scale
                        # Otherwise, user indices as scale
                        if not isinstance(req_attr['MCA_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['MCA_Scale'])
                            scale = np.array(f[p])
                            req_data[f"{req}_scale"] = scale
                        else:
                            req_data[f"{req}_scale"] = np.arange(0,np.shape(data)[1])
                            
                        # Apply 2d/1d normalization among first axis, if requested
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            data = detector_norm(data, norm_by)
                            
                        req_data[f"{req}"] = data
                        
                    # This is for 3d STACK data
                    elif req_attr['type'] == 'STACK':
                        p = self.config.get_path(self.scan,req_attr['STACK_Path'])
                        data = np.array(f[p])
                                              
                        # Get the MCA scale or use points
                        if not isinstance(req_attr['MCA_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['MCA_Scale'])
                            scale = np.array(f[p])
                            req_data[f"{req}_scale1"] = scale
                        else:
                            req_data[f"{req}_scale1"] = np.arange(0,np.shape(data)[1])
                            
                        # Get the image scale or use points
                        if not isinstance(req_attr['STACK_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['STACK_Scale'])
                            scale = np.array(f[p])
                            req_data[f"{req}_scale2"] = scale
                        else:
                            req_data[f"{req}_scale2"] = np.arange(0,np.shape(data)[2])
                            
                        # Apply 3d/1d normalization among first axis, if requested
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            data = stack_norm(data, norm_by)
                        
                        req_data[f"{req}"] = data
                        
                    else:
                        raise Exception("Undefined type.")
            else:
                # If no alias defined, try to get data from SCA folder(s)
                try:
                    req_data[req] = self.sca_data[req].dropna().to_numpy()
                except:
                    raise Exception("Data stream undefined")
                    
        return req_data

class ScanInfo:

    def __init__(self, config, file, keys):
        """Load specific scan from data file and retrieve meta data

            Parameters
            ----------
            config: dict
                h5 configuration
            file: string
                file name
            keys: list, string
                list of key paths to the desired information
        """
        
        # Try opening hdf5 container
        try:
            with h5py.File(file, 'r') as f:
                # Create dictionary: key->scan_number->data
                info_dict = dict()

                #  Ensure type of keys is list
                if not isinstance(keys, list):
                    keys = [keys]

                # Iterate over all keys
                # Store these keys as keys for the info_dict
                for key in keys:
                    info_dict[key] = dict()
                    for k in f.keys():
                        try:
                            # Determine scan number from all top-level group entries in the h5 file
                            skey = int(config.get_h5scan(k))

                            # Append the data for the specific scan to the info_dict->dict
                            # Try to decode information as utf-8 if string
                            try:
                                info_dict[key][skey] = f[f'{k}/{key}'][()].decode("utf-8")
                            # else, stored as data array of length 1, extract first entry
                            except AttributeError:
                                # some magic to get the stored array entry
                                entry = f[f'{k}/{key}'][()]
                                if isinstance(entry, np.ndarray) and len(entry)==1:
                                    entry = entry[0]
                                info_dict[key][skey] = entry
                        except:
                            pass

        except:
            raise KeyError("Error opening and processing file.")

        self.info_dict = info_dict