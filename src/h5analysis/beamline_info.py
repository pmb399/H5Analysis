"""Internal functions to retrieve and handle meta data"""

# Scientific modules
import numpy as np
import pandas as pd

# Scan Info Loader
from .ReadData import ScanInfo

# Util functions
from .data_1d import apply_kwargs_1d
from .util import invert_dict, clean_beamline_info_dict
from collections import defaultdict
import warnings


def load_beamline(config, file, key, average=True, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, legend_item=None, twin_y=False):
    """Load beamline meta data.

        Parameters
        ----------
        config: dict,
            h5 data configuration
        file: string
            file name
        key: string
            path to the meta data of interest
        **kwargs:
            average: Boolean
                determines if array of values or their average is reported
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
            legend_item: string
                Name for legend
            twin_y: boolean
                supports a second y-axis on the right-hand side
                
        Returns
        -------
        data: dict
            dict with meta data
    """

    # Init the data storage
    class beamline_object:
        def __init__(self):
            pass

    data = dict()
    data[0] = beamline_object()
    data[0].xlabel = 'Scan Number'
    data[0].xaxis_label = ['Scan Number']
    data[0].ylabel = key
    data[0].filename = file
    data[0].twin_y = twin_y

    # Get legend items
    if legend_item != None:
        data[0].legend = legend_item
    else:
        data[0].legend = f"{config.index}-{file}-{key}"

    # Get the scan information from file
    infoObj = ScanInfo(config, file, key, average)
    info = infoObj.info_dict

    # Parse data to numpy array
    keys = list()
    values = list()
    for key, value in info[key].items():
        keys.append(key)
        if not isinstance(value, tuple):
            values.append(np.average((value)))
        else:
            values.append(value[0])  # return the average

    info_array = np.zeros((len(keys), 2))
    info_array[:, 0] = np.array(keys, dtype='float')
    info_array[:, 1] = np.array(values, dtype='float')

    # Store the data
    data[0].x_stream = info_array[:, 0]
    data[0].y_stream = info_array[:, 1]
    data[0].scan = 0

    # Apply kwargs
    data[0].x_stream,data[0].y_stream = apply_kwargs_1d(data[0].x_stream,data[0].y_stream,norm,xoffset,xcoffset,yoffset,ycoffset,[None, None, None],None,None)

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


def get_spreadsheet(config, file, columns, average=True):
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
        average: Boolean
            determines if array of values or their average is reported

    """

    # Store all dictionary values in list
    # Keep track which values to concatenate
    key_list = list()
    concat_list = list()
    for idx, key in enumerate(list(columns.values())):
        if isinstance(key, str):
            key_list.append(key)
        elif isinstance(key, tuple):
            # these will get concatenated
            concat_list.append(key)
            for k in key:
                # but append to key_list first, to get individual data
                key_list.append(k)
        elif isinstance(key, list):
            if len(key) == 2:
                key_entry = key[0]
            else:
                raise Exception(
                    f"Wrong number of arguments specified in {key}.")
            if isinstance(key_entry, str):
                key_list.append(key_entry)
            elif isinstance(key_entry, tuple):
                # these will get concatenated
                concat_list.append(key_entry)
                for k in key_entry:
                    # but append to key_list first, to get individual data
                    key_list.append(k)
        else:
            raise Exception("Data type not understood.")

    # Get the meta data for all specified dict keys
    infoObj = ScanInfo(config, file, key_list, average=average)
    info = infoObj.info_dict

    concat_info = dict()
    # Combine columns, i.e. keys in dict
    for idx, concat in enumerate(concat_list):
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
    for key, value in columns.items():
        if isinstance(value, str):
            data_info[value] = info[value]
        elif isinstance(value, tuple):
            data_info[value] = concat_info[value]
        elif isinstance(value, list):
            value_entry = value[0]
            if isinstance(value_entry, str):
                data_info[value_entry] = info[value_entry]
            elif isinstance(value_entry, tuple):
                data_info[value_entry] = concat_info[value_entry]
        else:
            raise Exception("Data type not understood.")

    # generate pandas data frame to store the entries
    # returns only first element from values if type is list
    clean_columns = clean_beamline_info_dict(columns)
    df = pd.DataFrame(data_info).rename(
        invert_dict(clean_columns), axis=1).fillna('')

    # Set row labels as scan and name index column
    df.set_axis(list(list(data_info.values())[0].keys()))
    df.index.name = 'Scan'

    # Apply rounding
    for header, decimal_info in columns.items():
        if isinstance(decimal_info, list):
            decimals = int(decimal_info[1])
            if isinstance(decimal_info[0], tuple):
                try:
                    df[header] = df[header].apply(
                        lambda x: apply_rounding_tuple(x, decimals))
                except:
                    pass
            else:
                try:
                    df[header] = df[header].apply(
                        lambda x: apply_rounding(x, decimals))
                except:
                    pass

    return df


def apply_rounding(item, decimals):
    """Apply rounding to meta data"""
    if isinstance(item, tuple):
        i1_rounded = np.round(item[0], decimals)
        ii_rounded = np.round(item[1], decimals)
        if_rounded = np.round(item[2], decimals)
        return f"({i1_rounded},{ii_rounded},{if_rounded})"
    else:
        return np.round(item, decimals)


def apply_rounding_tuple(item, decimals):
    """Apply rounding to a meta data tuple"""
    contributions = item.split("; ")
    cont = [x for x in contributions if x != '']

    item = ""

    for x in cont:
        try:
            r = np.round(float(x), decimals)
            item += f"{r}; "
        except:
            try:
                strtup = x.split("(")[1].rstrip(")").split(',')
                t = [float(a) for a in strtup]
                r1 = np.round(t[0], decimals)
                ri = np.round(t[1], decimals)
                rf = np.round(t[2], decimals)
                item += f"({r1},{ri},{rf}); "
            except Exception as e:
                item += f"{x}; "
    return item
