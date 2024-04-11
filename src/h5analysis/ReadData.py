"""Tools to read in requested HDF5 data as object"""

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
        
        # open h5 file
        with h5py.File(self.file, 'r') as f:

            # Iterate over all requests
            for req in requisition:
                # If an alias is defined, use config file
                # Check whether we can find an alias
                if check_key_in_dict(req,self.config.h5dict):
            
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
                            req_data[req] = np.true_divide(data,norm_by,where=norm_by!=0)
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

                        if len(np.shape(data)) == 2:
                            data = data[np.newaxis,...]
                                              
                        # Get the MCA scale or use points
                        if not isinstance(req_attr['MCA_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['MCA_Scale'])
                            scale = np.array(f[p])
                            req_data[f"{req}_scale1"] = scale
                        else:
                            req_data[f"{req}_scale1"] = np.arange(0,np.shape(data)[1])

                        # With stack dimension (n,x,y)
                        # Always require the scale to be of dim (n,x) and (n,y)
                        if len(np.shape(req_data[f"{req}_scale1"])) == 1:
                            req_data[f"{req}_scale1"] = np.repeat(req_data[f"{req}_scale1"][None,...],np.shape(data)[0],0)
                            
                        # Get the image scale or use points
                        if not isinstance(req_attr['STACK_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['STACK_Scale'])
                            scale = np.array(f[p])
                            req_data[f"{req}_scale2"] = scale
                        else:
                            req_data[f"{req}_scale2"] = np.arange(0,np.shape(data)[2])

                        # With stack dimension (n,x,y)
                        # Always require the scale to be of dim (n,x) and (n,y)
                        if len(np.shape(req_data[f"{req}_scale2"])) == 1:
                            req_data[f"{req}_scale2"] = np.repeat(req_data[f"{req}_scale2"][None,...],np.shape(data)[0],0)
                            
                        # Apply 3d/1d normalization among first axis, if requested
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            data = stack_norm(data, norm_by)
                        
                        req_data[f"{req}"] = data
                        
                    else:
                        raise Exception("Undefined type.")
                    
                # If no alias defined, try to get data from SCA folder(s)
                else:
                    # In case no data stream but only ROI is given (for "None" ROIs)
                    if req != '':
                        # Check if any SCA folder paths are specified
                        # Makes all 1d data in specified folder(s) accessible
                        if self.config.sca_folders != list():
                                for path in self.config.sca_folders:
                                    has_data = False
                                    # Get the full h5 group path
                                    p = self.config.get_path(self.scan,path)
                                    try:
                                        req_data[req] = np.array(f[f'{p}/{req}'])
                                        has_data = True
                                        break # Need to break after finding stream
                                    except:
                                        pass

                                if has_data == False:
                                    raise Exception(f"Data stream {req} undefined")

                        else:
                            raise Exception(f"Data stream {req} undefined")
                    else:
                        # Define fake array when no stream is given and req is empty string
                        req_data[req] = [0]
                    
        return req_data

class ScanInfo:
    """Standard data class to store h5 meta data"""

    def __init__(self, config, file, keys, average):
        """Load specific scan from data file and retrieve meta data

            Parameters
            ----------
            config: dict
                h5 configuration
            file: string
                file name
            keys: list, string
                list of key paths to the desired information
            average: Boolean
                determines if array of values or their average is reported
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
                                item = f[f'{k}/{key}'][()].decode("utf-8")
                                try:
                                    info_dict[key][skey] = float(item)
                                except:
                                    info_dict[key][skey] = item
                            # else, stored as data array of length 1, extract first entry
                            except AttributeError:
                                # some magic to get the stored array entry
                                entry = f[f'{k}/{key}'][()]
                                if isinstance(entry, np.ndarray) and len(entry)==1:
                                    entry = entry[0]
                                else:
                                    if average == True:
                                        entry = (np.average(entry),entry.min(),entry.max())

                                info_dict[key][skey] = entry
                        except:
                            pass

        except Exception as e:
            raise Exception(f"Error opening and processing file. {e}")

        self.info_dict = info_dict