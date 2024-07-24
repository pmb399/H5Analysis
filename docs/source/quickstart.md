# Quickstart

## Start a jupyter notebook instance

Open a terminal window (python shell in Windows) and launch your local jupyter installation with

```
$ jupyter notebook
```

## Imports

Before you start, you will need to import the required h5analysis package, and enable bokeh plotting.

```
# Setup necessary inputs

## Import the loader classes
from h5analysis.LoadData import *
from h5analysis.MathData import *

## Import the configuration class
from h5analysis.config import h5Config

## Enable bokeh plotting within the jupyter notebook
from bokeh.io import show, output_notebook
output_notebook(hide_banner=True)
```

## Data File Format configuration

All data loaders require a proper configuration variable which is beamline/data format specific. An example configuration for the CLS REIXS beamline may look like this:

```
config = h5Config()

config.key("SCAN_{scan:03d}",'scan')
config.sca_folder('Data')

config.sca('Mono Energy','Data/beam')
config.sca('Mesh Current','Data/i0')
config.sca('Sample Current','Data/tey')
config.sca('TEY','Data/tey','Data/i0')
config.sca("MCP Energy", 'Data/mcpMCA_scale')
config.sca("SDD Energy", 'Data/sddMCA_scale')
config.sca("XEOL Energy", 'Data/xeolMCA_scale')

config.mca('SDD','Data/sddMCA','Data/sddMCA_scale',None)
config.mca('MCP','Data/mcpMCA','Data/mcpMCA_scale',None)
config.mca('XEOL','Data/xeolMCA','Data/xeolMCA_scale',None)

config.stack('mcpSTACK','Data/mcp_a_img','Data/mcpMCA_scale',None,None)
```

You will need to create a ```h5Config()``` object, add the appropriate formating to access various scan groups within the h5 file, specify folder with 1d single channel analyzer (SCA) data (if existent), and may add 2d multi channel analyzer (MCA) data streams, as well as 3d image stack data. We define the alias, alongside the data location within the h5 file, alongside various key-word arguments such as scales, normalization channels, labels, etc. which are documented in the API section.