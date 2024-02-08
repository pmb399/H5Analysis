# Scientific modules
import numpy as np
import pandas as pd

# Scan Info Loader
from .ReadData import ScanInfo

# Util functions
from .simplemath import apply_offset
from .util import invert_dict, clean_beamline_info_dict
from collections import defaultdict

def load_beamline(config, file, key, average=False, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, legend_item=None):
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
            average: Boolean
                determines if array of values or their average is reported
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
    infoObj = ScanInfo(config,file,key,average)
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


def get_single_beamline_value(config, file, keys, *args, average=False):
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
        kwargs:
            average: Boolean
                determines if array of values or their average is reported
    """

    # Get the meta data
    infoObj = ScanInfo(config, file, keys, average)
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


def get_spreadsheet(config, file, average=True, columns=None):
    """Generate spreadsheet with meta data from h5 file.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        average: Boolean
            determines if array of values or their average is reported
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

    # Store all dictionary values in list
    # Keep track which values to concatenate
    key_list = list()
    concat_list = list()
    for idx,key in enumerate(list(columns.values())):
        if isinstance(key,str):
            key_list.append(key)
        elif isinstance(key,tuple):
            # these will get concatenated
            concat_list.append(key)
            for k in key:
                # but append to key_list first, to get individual data
                key_list.append(k)
        elif isinstance(key,list):
            if len(key) == 2:
                key_entry = key[0]
            else:
                raise Exception(f"Wrong number of arguments specified in {key}.")
            if isinstance(key_entry,str):
                key_list.append(key_entry)
            elif isinstance(key_entry,tuple):
                # these will get concatenated
                concat_list.append(key_entry)
                for k in key_entry:
                    # but append to key_list first, to get individual data
                    key_list.append(k) 
        else:
            raise Exception("Data type not understood.")

    # Get the meta data for all specified dict keys
    infoObj = ScanInfo(config, file,key_list,average=average)
    info = infoObj.info_dict

    concat_info = dict()
    # Combine columns, i.e. keys in dict
    for idx,concat in enumerate(concat_list):
        # Store all dicts with keys in single concat list
        concat_dicts = list()
        # Generate new defaultdict with combined results
        new_dict = defaultdict(str)

        # Get all dicts for concat tuple
        for key in concat:
            concat_dicts.append(info[key])

        # Populate combined dict
        for d in tuple(concat_dicts):
            for key, value in d.items():
                new_dict[key] += f"{str(value)}; "

        # Add combined dict to "global" info dict
        concat_info[concat] = new_dict

    data_info = dict()
    for key,value in columns.items():
        if isinstance(value,str):
            data_info[value] = info[value]
        elif isinstance(value,tuple):
            data_info[value] = concat_info[value]
        elif isinstance(value,list):
            value_entry = value[0]
            if isinstance(value_entry,str):
                data_info[value_entry] = info[value_entry]
            elif isinstance(value_entry,tuple):
                data_info[value_entry] = concat_info[value_entry]
        else:
            raise Exception("Data type not understood.")
        
    # generate pandas data frame to store the entries
    clean_columns = clean_beamline_info_dict(columns) # returns only first element from values if type is list
    df = pd.DataFrame(data_info).rename(invert_dict(clean_columns), axis=1).fillna('')

    # Set row labels as scan and name index column
    df.set_axis(list(list(data_info.values())[0].keys()))
    df.index.name = 'Scan'

    # Apply rounding
    for header,decimal_info in columns.items():
        if isinstance(decimal_info,list):
            decimals = int(decimal_info[1])
            if isinstance(decimal_info[0],tuple):
                df[header] = df[header].apply(lambda x: apply_rounding_tuple(x,decimals))
            else:
                df[header] = df[header].apply(lambda x: apply_rounding(x,decimals)) 

    return df

def apply_rounding(item, decimals):
    if isinstance(item,tuple):
        i1_rounded = np.round(item[0],decimals)
        ii_rounded = np.round(item[1],decimals)
        if_rounded = np.round(item[2],decimals)
        return f"({i1_rounded},{ii_rounded},{if_rounded})"
    else:
        return np.round(item,decimals)

def apply_rounding_tuple(item,decimals):
    contributions = item.split("; ")
    cont = [x for x in contributions if x != '']

    item = ""

    for x in cont:
        try:
            r = np.round(float(x),decimals)
            item += f"{r}; "
        except:
            try:
                strtup = x.split("(")[1].rstrip(")").split(',')
                t = [float(a) for a in strtup]
                r1 = np.round(t[0],decimals)
                ri = np.round(t[1],decimals)
                rf = np.round(t[2],decimals)
                item += f"({r1},{ri},{rf}); "
            except Exception as e:
                item += f"{x}; "
    return item