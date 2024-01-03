import pandas as pd
import h5py
import numpy as np
from .util import check_key_in_dict
from .readutil import detector_norm, stack_norm

class Data:
    """Standard data class to store h5 files"""

    def __init__(self,config,file,scan):
        self.config = config
        self.file = file
        self.scan = scan
        
        # Makes all 1d data in specified folder(s) accessible
        if self.config.sca_folders != list():
            self.sca_data = pd.DataFrame()
            sca_series = list()
            sca_headers = list()

            with h5py.File(file, 'r') as f:

                # Populate a pandas dataframe with all SCA data
                try:
                    for path in self.config.sca_folders:
                        p = config.get_path(scan,path)
                        for entry in f[p]:
                            if len(f[f'{p}/{entry}'].shape) == 1:
                                sca_series.append(pd.Series(np.array(f[f'{p}/{entry}'])))
                                sca_headers.append(str(entry))
                                
                    self.sca_data = pd.DataFrame(sca_series).transpose(copy=True)
                    self.sca_data.columns = sca_headers
                                
                except Exception as e:
                    raise Exception(e)
                    
    def Scan(self,requisition):
        req_data = dict()
        
        if not isinstance(requisition,list):
            requisition = [requisition]
        
        for req in requisition:
            
            # If an alias is defined, use config file
            if check_key_in_dict(req,self.config.h5dict):
                with h5py.File(self.file, 'r') as f:
        
                    req_attr = self.config.h5dict[req]
            
                    if req_attr['type'] == 'SCA':
                        p = self.config.get_path(self.scan,req_attr['SCA_Path'])
                        data = np.array(f[p])
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            req_data[req] = np.true_divide(data,norm_by)
                        else:
                            req_data[req] = data
                            
                    elif req_attr['type'] == 'MCA':
                        p = self.config.get_path(self.scan,req_attr['MCA_Path'])
                        data = np.array(f[p])
                        
                        if not isinstance(req_attr['MCA_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['MCA_Scale'])
                            scale = np.array(f[p])
                            
                            req_data[f"{req}_scale"] = scale

                        else:
                            req_data[f"{req}_scale"] = np.arange(0,np.shape(data)[1])
                            
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            data = detector_norm(data, norm_by)
                            
                        req_data[f"{req}"] = data
                        
                    elif req_attr['type'] == 'STACK':
                        p = self.config.get_path(self.scan,req_attr['STACK_Path'])
                        data = np.array(f[p])
                                              
                        if not isinstance(req_attr['MCA_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['MCA_Scale'])
                            scale = np.array(f[p])
                            
                            req_data[f"{req}_scale1"] = scale

                        else:
                            req_data[f"{req}_scale1"] = np.arange(0,np.shape(data)[1])
                            
                        if not isinstance(req_attr['STACK_Scale'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['STACK_Scale'])
                            scale = np.array(f[p])
                            
                            req_data[f"{req}_scale2"] = scale

                        else:
                            req_data[f"{req}_scale2"] = np.arange(0,np.shape(data)[2])
                            
                        if not isinstance(req_attr['norm_by'],type(None)):
                            p = self.config.get_path(self.scan,req_attr['norm_by'])
                            norm_by = np.array(f[p])
                            data = stack_norm(data, norm_by)
                            
                        
                        req_data[f"{req}"] = data
                        
                    else:
                        raise Exception("Undefined type.")
            else:
                # If no alias defined, try to get data from SCA folder
                try:
                    req_data[req] = self.sca_data[req].dropna().to_numpy()
                except:
                    raise Exception("Data stream undefined")
                    
        return req_data

class ScanInfo:

    def __init__(self, config, file, keys):
        """Load one specific scan from specified data file

        Parameters
        ----------
        keys : list of key paths to the desired information
        """
        # Try opening hdf5 container
        try:
            with h5py.File(file, 'r') as f:
                # Create dictionary for scan numbers
                info_dict = dict()

                if not isinstance(keys, list):
                    keys = [keys]

                for key in keys:
                    info_dict[key] = dict()
                    for k in f.keys():
                        try:
                            skey = int(config.get_h5scan(k))
                            try:
                                info_dict[key][skey] = f[f'{k}/{key}'][()].decode("utf-8")
                            except AttributeError:
                                entry = f[f'{k}/{key}'][()]
                                if isinstance(entry, np.ndarray) and len(entry)==1:
                                    entry = entry[0]
                                info_dict[key][skey] = entry
                        except:
                            pass

        except:
            raise KeyError("Error opening and processing file.")

        self.info_dict = info_dict