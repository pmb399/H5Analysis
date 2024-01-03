import numpy as np
import pandas as pd
from .ReadData import ScanInfo
from .simplemath import apply_offset
from .util import invert_dict


def load_beamline(config, file, key, norm=False, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None, legend_item=None):
    class beamline_object:
        def __init__(self):
            pass

    data = dict()
    data[0] = beamline_object()

    infoObj = ScanInfo(config,file,key)
    info = infoObj.info_dict
    info_array = np.array(list(info[key].items()), dtype='float')

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
    infoObj = ScanInfo(config, file, keys)
    info = infoObj.info_dict

    if not isinstance(keys, list):
        keys = [keys]

    for entry in keys:
        print('=====  ', entry, '  =====')

        for k, v in info[entry].items():
            if k in args:
                print(f"Scan {k} -", v)

        print('====================')


def get_spreadsheet(config, file, columns=None):
    if columns == None:
        columns = dict()

        #columns['Sample'] = 'command'
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
        #columns['Comment'] = 'command'
        columns['Status'] = 'status'

    infoObj = ScanInfo(config, file,list(columns.values()))
    info = infoObj.info_dict
    return pd.DataFrame(info).rename(invert_dict(columns), axis=1)
