import re
from itertools import groupby
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from collections.abc import MutableMapping

#########################################################################################


def all_list_entries_equal(iterable):
    """Checks if all entries in a list are equal"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

#########################################################################################


def doesMatchPattern(string, patterns=[]):
    """Checks if a string matches specific patterns"""
    for p in patterns:

        # skip empty patterns
        if not p:
            continue

        if re.search(p, string, re.I):
            return True

    return False

#########################################################################################


def get_roi(roi):
    """Gets the roi when ':' separated"""
    try:
        roi_low = float(roi.split(":")[0])
        roi_high = float(roi.split(":")[1])
    except:
        roi_low = float(roi)
        roi_high = float(roi)

    return check_idx(roi_low, roi_high)

#########################################################################################


def check_idx(idx_low, idx_high):
    """Check the index of an array. Add +1 to allow slicing."""
    if idx_low == idx_high:
        idx_high = idx_low+1

    return idx_low, idx_high

#########################################################################################


def check_key_in_dict(key, dic):
    """Checks if a specific key is in a dictionary"""
    for k, v in dic.items():
        if key == k:
            return True

#########################################################################################


def flatten(d):
    """Flattens a dictionary of dicts to one dictionary"""
    items = []
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            items.extend(flatten(v).items())
        else:
            items.append((k, v))
    return dict(items)

#########################################################################################


def invert_dict(my_map):
    return {v: k for k, v in my_map.items()}

#########################################################################################
# Palette for Bokeh Plots


COLORP = ['#d60000', '#8c3bff', '#018700', '#00acc6', '#e6a500', '#ff7ed1', '#6b004f', '#573b00', '#005659', '#15e18c', '#0000dd', '#a17569', '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c', '#ff7752', '#004b00', '#8e7b01', '#f2007b', '#8eba00', '#a57bb8', '#5901a3', '#e2afaf', '#a03a52', '#a1c8c8', '#9e4b00', '#546744', '#bac389', '#5e7b87',
          '#60383b', '#8287ff', '#380000', '#e252ff', '#2f5282', '#7ecaff', '#c4668e', '#008069', '#919eb6', '#cc7407', '#7e2a8e', '#00bda3', '#2db152', '#4d33ff', '#00e400', '#ff00cd', '#c85748', '#e49cff', '#1ca1ff', '#6e70aa', '#c89a69', '#77563b', '#03dae6', '#c1a3c3', '#ff6989', '#ba00fd', '#915280', '#9e0174', '#93a14f', '#364424', '#af6dff', '#596d00',
          '#ff3146', '#828056', '#006d2d', '#8956af', '#5949a3', '#773416', '#85c39a', '#5e1123', '#d48580', '#a32818', '#0087b1', '#ca0044', '#ffa056', '#eb4d00', '#6b9700', '#528549', '#755900', '#c8c33f', '#91d370', '#4b9793', '#4d230c', '#60345b', '#8300cf', '#8a0031', '#9e6e31', '#ac8399', '#c63189', '#015438', '#086b83', '#87a8eb', '#6466ef', '#c35dba',
          '#019e70', '#805059', '#826e8c', '#b3bfda', '#b89028', '#ff97b1', '#a793e1', '#698cbd', '#4b4f01', '#4801cc', '#60006e', '#446966', '#9c5642', '#7bacb5', '#cd83bc', '#0054c1', '#7b2f4f', '#fb7c00', '#34bf00', '#ff9c87', '#e1b669', '#526077', '#5b3a7c', '#eda5da', '#ef52a3', '#5d7e69', '#c3774f', '#d14867', '#6e00eb', '#1f3400', '#c14103', '#6dd4c1',
          '#46709e', '#a101c3', '#0a8289', '#afa501', '#a55b6b', '#fd77ff', '#8a85ae', '#c67ee8', '#9aaa85', '#876bd8', '#01baf6', '#af5dd1', '#59502a', '#b5005e', '#7cb569', '#4985ff', '#00c182', '#d195aa', '#a34ba8', '#e205e2', '#16a300', '#382d00', '#832f33', '#5d95aa', '#590f00', '#7b4600', '#6e6e31', '#335726', '#4d60b5', '#a19564', '#623f28', '#44d457',
          '#70aacf', '#2d6b4d', '#72af9e', '#fd1500', '#d8b391', '#79893b', '#7cc6d8', '#db9036', '#eb605d', '#eb5ed4', '#e47ba7', '#a56b97', '#009744', '#ba5e21', '#bcac52', '#87d82f', '#873472', '#aea8d1', '#e28c62', '#d1b1eb', '#36429e', '#3abdc1', '#669c4d', '#9e0399', '#4d4d79', '#7b4b85', '#c33431', '#8c6677', '#aa002d', '#7e0175', '#01824d', '#724967',
          '#727790', '#6e0099', '#a0ba52', '#e16e31', '#c46970', '#6d5b95', '#a33b74', '#316200', '#87004f', '#335769', '#ba8c7c', '#1859ff', '#909101', '#2b8ad4', '#1626ff', '#21d3ff', '#a390af', '#8a6d4f', '#5d213d', '#db03b3', '#6e56ca', '#642821', '#ac7700', '#a3bff6', '#b58346', '#9738db', '#b15093', '#7242a3', '#878ed1', '#8970b1', '#6baf36', '#5979c8',
          '#c69eff', '#56831a', '#00d6a7', '#824638', '#11421c', '#59aa75', '#905b01', '#f64470', '#ff9703', '#e14231', '#ba91cf', '#34574d', '#f7807c', '#903400', '#b3cd00', '#2d9ed3', '#798a9e', '#50807c', '#c136d6', '#eb0552', '#b8ac7e', '#487031', '#839564', '#d89c89', '#0064a3', '#4b9077', '#8e6097', '#ff5238', '#a7423b', '#006e70', '#97833d', '#dbafc8']
#########################################################################################


def to_quanty(file):
    """Manipulates data file such that it can be used as input for Quanty Fitting program.

    Parameters
    ----------
    file : string
        Specify path to exported file.

    """
    # Read the data file

    df = pd.read_csv(file, header=1)
    minimum = list()
    maximum = list()
    diff = list()
    interpObj = list()
    # Start interpolation process to common energy loss axis
    for i, name in enumerate(df.columns):
        if name.endswith('MCP Energy'):
            minimum.append(df[name].min())
            maximum.append(df[name].max())
            diff.append(np.diff(df[name]).min())
            # Interpolate all data
            interpObj.append(interp1d(np.array(df[name]), np.array(
                df.iloc[:, df.columns.get_loc(name)+1]), fill_value='extrapolate'))

    # Calculate the number of points and generate linear space
    numPoints = int(np.ceil((np.array(maximum).max() -
                    np.array(minimum).min())/abs(np.array(diff).min())))
    linspace = np.linspace(np.array(minimum).min(), np.array(
        maximum).max(), numPoints, endpoint=True)

    # Create new pandas data frame
    dfint = pd.DataFrame(linspace)

    # Evaluate all interp objects on new lin space
    # and append to new data frame
    for i, obj in enumerate(interpObj):
        dfint[i+1] = obj(linspace)

    # Convert data frame to numpy and export
    npreturn = dfint.to_numpy()
    np.savetxt(f"{file}_Quanty.txt", npreturn)
