# Scientific Modules
import numpy as np
import pandas as pd
import h5py

# Scan analysis utils
from .readutil import mca_roi, detector_norm, subtract_background
from .simplemath import apply_offset, grid_data2d

# Spec Config
from .spec_config import get_REIXSconfig, get_h5key, get_h5scan

# Utilities
import os
import warnings

## Every scan will be an instance of a class ##

#########################################################################################


class REIXS:
    """data object HDF5

        Parameters
        ----------
        header_file : string
            Give name of header file with extension
    """

    def __init__(self, file, scan):
        self.file = file

        # Read in parameter/variable config (either external from os.env or internal default)
        self.REIXSconfig = get_REIXSconfig()
        self.group = get_h5key(scan)

        with h5py.File(file, 'r') as f:

            # Populate a pandas dataframe with all SCA data
            self.sca_data = pd.DataFrame()
            try:
                for entry in f[f'{self.group}/{self.REIXSconfig["HDF5_sca_data"]}']:
                    if len(f[f'{self.group}/{self.REIXSconfig["HDF5_sca_data"]}/{entry}'].shape) == 1 and len(f[f'{self.group}/{self.REIXSconfig["HDF5_sca_data"]}/{entry}']) == len(f[f'{self.group}/{self.REIXSconfig["HDF5_sca_data"]}/epoch']):
                        self.sca_data[str(entry)] = np.array(
                            f[f'{self.group}/{self.REIXSconfig["HDF5_sca_data"]}/{entry}'])
            except Exception as e:
                warnings.warn(
                    f"Could not load SCAs from HDF5 container. {type(e)}: {e}")

    def Scan(self, request, roi, kwargs=dict()):

        # Try opening hdf5 container
        with h5py.File(self.file, 'r') as f:
                               
            # Load data for the requested alias
                            
            if self.REIXSconfig[request]['type'] == 'SCA':
                data = np.array(f[f"{self.group}/{self.REIXSconfig[request]['SCA_path']}"])

                if not isinstance(self.REIXSconfig[request]['norm_by'],type(None)):
                    norm_by = np.array(f[f"{self.group}/{self.REIXSconfig[request]['norm_by']}"])
                    self.data = np.true_divide(data,norm_by)
                else:
                    self.data = data

            elif self.REIXSconfig[request]['type'] == 'MCA':
                mca = np.array(f[f"{self.group}/{self.REIXSconfig[request]['MCA_path']}"])
                roi_scale = np.array(f[f"{self.group}/{self.REIXSconfig[request]['ROI_scale']}"])

                if 'background' in kwargs and kwargs['background'] != None:
                    bg_group = get_h5key(kwargs['background'])
                    bg = np.array(f[f"{bg_group}/{self.REIXSconfig[request]['MCA_path']}"])

                    mca = subtract_background(mca,bg)

                if not isinstance(self.REIXSconfig[request]['norm_by'],type(None)):
                    norm_by = np.array(f[f"{self.group}/{self.REIXSconfig[request]['norm_by']}"])
                    mca = detector_norm(mca, norm_by)
                
                self.data = mca_roi(mca,roi_scale,roi,self.REIXSconfig[request]['summation_axis'])

            elif self.REIXSconfig[request]['type'] == 'IMG':
                mca = np.array(f[f"{self.group}/{self.REIXSconfig[request]['MCA_path']}"])
                self.data_scale = np.array(f[f"{self.group}/{self.REIXSconfig[request]['Data_scale']}"])

                if 'background' in kwargs and kwargs['background'] != None:
                    bg_group = get_h5key(kwargs['background'])
                    bg = np.array(f[f"{bg_group}/{self.REIXSconfig[request]['MCA_path']}"])

                    mca = subtract_background(mca,bg) 

                if not isinstance(self.REIXSconfig[request]['norm_by'],type(None)):
                    norm_by = np.array(f[f"{self.group}/{self.REIXSconfig[request]['norm_by']}"])
                    self.mca = detector_norm(mca, norm_by)
                else:
                    self.mca = mca

            else:
                raise UserWarning("Type not implemented.")


class ScanInfo:

    def __init__(self,file, keys):
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
                            skey = int(get_h5scan(k))
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