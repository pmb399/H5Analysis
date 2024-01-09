# Scientific modules
import numpy as np
import pandas as pd

# Scan Info Loader
from .ReadData import ScanInfo

# Util functions
from .simplemath import apply_offset
from .util import invert_dict

def load_beamline(config, file, key, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, legend_item=None):
    """Load beamline meta data.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        key: string
            path to the meta data of interest
        kwargs: See Load1d class, but additional
            xoffset: fitting offset
            yoffset: fitting offset
            xcoffset: constant offset
            ycoffset: constant offset
            legend_item: string

        Returns
        -------
        data: dict
    """

    # Init the data storage
    class beamline_object:
        def __init__(self):
            pass

    data = dict()
    data[0] = beamline_object()
    data[0].xlabel = 'Scan Number'
    data[0].ylabel = key
    data[0].filename = file

    # Get the scan information from file
    infoObj = ScanInfo(config,file,key)
    info = infoObj.info_dict

    # Parse data to numpy array
    info_array = np.array(list(info[key].items()), dtype='float')

    # Store the data
    data[0].x_stream = info_array[:, 0]
    data[0].y_stream = info_array[:, 1]
    data[0].scan = key

    # Get legend items
    if legend_item != None:
        data[0].legend = legend_item
    else:
        data[0].legend = f"{file} - {key}"

    # Apply normalization to [0,1]
    if norm == True:
        data[0].y_stream = np.interp(
            data[0].y_stream, (data[0].y_stream.min(), data[0].y_stream.max()), (0, 1))

    # Apply offset to x and y-stream
    data[0].x_stream = apply_offset(data[0].x_stream, xoffset, xcoffset)
    data[0].y_stream = apply_offset(data[0].y_stream, yoffset, ycoffset)

    return data


def get_single_beamline_value(config, file, keys, *args):
    """Load beamline meta data.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        keys: string, list
            path to the meta data of interest
        args: int
            scan numbers, comma separated
    """

    # Get the meta data
    infoObj = ScanInfo(config, file, keys)
    info = infoObj.info_dict

    # Check type of key, need it as list
    if not isinstance(keys, list):
        keys = [keys]

    # Print the results to screen
    for entry in keys:
        print('=====  ', entry, '  =====')

        for k, v in info[entry].items():
            if k in args:
                print(f"Scan {k} -", v)

        print('====================')


def get_spreadsheet(config, file, columns=None):
    """Generate spreadsheet with meta data from h5 file.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        columns: dict
            Specify column header and h5 data path to meta datam i.e.
                columns = dict()
                columns['Sample Stage horz'] = 'Endstation/Motors/ssh
                ...
    """
        
    if columns == None:
        columns = dict()

        columns['Command'] = 'command'
        columns['Sample Stage (ssh)'] = 'Endstation/Motors/ssh'
        columns['Sample Stage (ssv)'] = 'Endstation/Motors/ssv'
        columns['Sample Stage (ssd)'] = 'Endstation/Motors/ssd'
        columns['Spectrometer (XES dist)'] = 'Endstation/Motors/spd'
        columns['Spectrometer (XES angl)'] = 'Endstation/Motors/spa'
        columns['Flux 4-Jaw (mm)'] = 'Beamline/Apertures/4-Jaw_2/horz_gap'
        columns['Mono Grating'] = '/Beamline/Monochromator/grating'
        columns['Mono Mirror'] = '/Beamline/Monochromator/mirror'
        columns['Polarization'] = 'Beamline/Source/EPU/Polarization'
        columns['Status'] = 'status'

    # Get the meta data for all specified dict keys
    infoObj = ScanInfo(config, file,list(columns.values()))
    info = infoObj.info_dict

    # generate pandas data frame to store the entries
    return pd.DataFrame(info).rename(invert_dict(columns), axis=1)
