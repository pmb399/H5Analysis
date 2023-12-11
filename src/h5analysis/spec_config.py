import os
import ast
import parse

h5struct = "SCAN_{scan:03d}"

def get_h5key(scan):
    return h5struct.format(scan=scan)

def get_h5scan(key):
    return parse.parse(h5struct,key)['scan']

def get_REIXSconfig():
    REIXSconfig = dict()

    REIXSconfig["HDF5_sca_data"] = 'Data'
    REIXSconfig["HDF5_mono_energy"] = "Data/beam"
    REIXSconfig["HDF5_mesh_current"] = "Data/i0"
    REIXSconfig["HDF5_sample_current"] = "Data/tey"

    dataDict = dict()
    dataDict['type'] = 'SCA'
    dataDict['SCA_path'] = REIXSconfig["HDF5_mono_energy"]
    dataDict['norm_by'] = None
    REIXSconfig["Mono Energy"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'SCA'
    dataDict['SCA_path'] = 'Data/mcpMCA_scale'
    dataDict['norm_by'] = None
    REIXSconfig["MCP Energy"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'SCA'
    dataDict['SCA_path'] = 'Data/xeolMCA_scale'
    dataDict['norm_by'] = None
    REIXSconfig["XEOL Energy"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'SCA'
    dataDict['SCA_path'] = REIXSconfig["HDF5_sample_current"]
    dataDict['norm_by'] = REIXSconfig["HDF5_mesh_current"]
    REIXSconfig["TEY"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'SCA'
    dataDict['SCA_path'] = REIXSconfig["HDF5_sample_current"]
    dataDict['norm_by'] = None
    REIXSconfig["Sample"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'SCA'
    dataDict['SCA_path'] = REIXSconfig["HDF5_mesh_current"]
    dataDict['norm_by'] = None
    REIXSconfig["Mesh"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'MCA'
    dataDict['MCA_path']  = 'Data/sddMCA'
    dataDict['ROI_scale'] = 'Data/sddMCA_scale'
    dataDict['summation_axis'] = 1
    dataDict['norm_by'] = REIXSconfig["HDF5_mesh_current"]
    REIXSconfig["PFY"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'MCA'
    dataDict['MCA_path']  = 'Data/sddMCA'
    dataDict['ROI_scale'] = REIXSconfig["HDF5_mono_energy"]
    dataDict['summation_axis'] = 0
    dataDict['norm_by'] = None
    REIXSconfig["XRF"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'MCA'
    dataDict['MCA_path']  = 'Data/mcpMCA'
    dataDict['ROI_scale'] = REIXSconfig["HDF5_mono_energy"]
    dataDict['summation_axis'] = 0
    dataDict['norm_by'] = None
    REIXSconfig["XES"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'MCA'
    dataDict['MCA_path']  = 'Data/mcpMCA'
    dataDict['ROI_scale'] = 'Data/mcpMCA_scale'
    dataDict['summation_axis'] = 1
    dataDict['norm_by'] = REIXSconfig["HDF5_mesh_current"]
    REIXSconfig["specPFY"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'MCA'
    dataDict['MCA_path']  = 'Data/xeolMCA'
    dataDict['ROI_scale'] = REIXSconfig["HDF5_mono_energy"]
    dataDict['summation_axis'] = 0
    dataDict['norm_by'] = None
    REIXSconfig["XEOL"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'MCA'
    dataDict['MCA_path']  = 'Data/xeolMCA'
    dataDict['ROI_scale'] = 'Data/xeolMCA_scale'
    dataDict['summation_axis'] = 1
    dataDict['norm_by'] = REIXSconfig["HDF5_mesh_current"]
    REIXSconfig["POY"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'IMG'
    dataDict['MCA_path']  = 'Data/mcpMCA'
    dataDict['Data_scale'] = 'Data/mcpMCA_scale'
    dataDict['norm_by'] = None
    REIXSconfig["MCP"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'IMG'
    dataDict['MCA_path']  = 'Data/sddMCA'
    dataDict['Data_scale'] = 'Data/sddMCA_scale'
    dataDict['norm_by'] = None
    REIXSconfig["SDD"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'IMG'
    dataDict['MCA_path']  = 'Data/xeolMCA'
    dataDict['Data_scale'] = 'Data/xeolMCA_scale'
    dataDict['norm_by'] = None
    REIXSconfig["CCD"] = dataDict

    dataDict = dict()
    dataDict['type'] = 'STACK'
    dataDict['STACK_path']  = 'Data/mcp_a_img'
    dataDict['Data_scale'] = 'Data/mcpMCA_scale'
    dataDict['Image_scale'] = None
    REIXSconfig["STACK_MCP"] = dataDict

    return REIXSconfig