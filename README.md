# CLS H5Analysis

This is a library to analyse, plot, and export HDF5 data. The package is meant to provide a framework to load data into jupyter and enable data interaction.

## Installation

Install the package from PyPi with the pip package manager. This is the recommended way to obtain a copy for your local machine and will install all required dependencies.

```
    $ pip install h5analysis
```

You will also need [Jupyter Notebook](https://github.com/jupyter) together with python 3 on your local machine.

In case that certain widgets aren't rendered properly, make sure to enable the appropriate jupyter extensions

```
    $ jupyter nbextension enable --py widgetsnbextension
```

## Running

Launch your local jupyter installation with

```
    $ jupyter notebook
```

## Examples

### Load the required module

Before you start, you will need to import the required h5analysis package, and enable bokeh plotting.

```
## Setup necessarry inputs
from h5analysis.LoadData import *
from h5analysis.config import h5Config
from bokeh.io import show, output_notebook
output_notebook(hide_banner=True)
```

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

### 1d plots

```
sca = Load1d()
sca.load(config,'FileName.h5','x_stream','y_stream',1,2,3,4)  # Loads multiple scans individually
sca.add(config,'FileName.h5','x_stream','y_stream',1,2,3,4)  # Adds multiple scans
sca.subtract(config,'FileName.h5','x_stream','y_stream',1,2,3,4,norm=False) # Subtracts scans from the first scan
sca.xlim(lower_lim,upper_lim) # Sets the horizontal axis plot region
sca.ylim(lower_lim,upper_lim) # Sets the vertical axis plot region
sca.plot_legend("pos string as per bokeh") # Determines a specific legend position
sca.vline(position) # Draws a vertical line
sca.hline(position) # Draws a horizontal line
sca.label(pos_x,pos_y,'Text') # Adds a label to the plot
sca.plot() # Plots the defined object
sca.exporter() # Exports the data by calling an exporter widget
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **x_stream** quantities include:
- All quantities contained in the sca folder(s) specified in the config
- All SCA specified in the config

4. Options for **y_stream** quantities include:
- All quantities contained in the sca folder(s) specified in the config
- All SCA specified in the config
- All MCA specified in the config with applied ROI
- All STACK specified in the config with two applied ROIs

5. List all scans to analyse (comma-separated)

6. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]
- _grid_x_ (Takes a list with three arguments to apply 1d interpolation gridding)
  e.g. grid_x = [Start Energy, Stop Energy, Delta]
- _savgol_ (Takes a list with two or three arguments to apply data smoothing and derivatives)
  e.g. savgol = [Window length, Polynomial order, deriavtive] as specified in the scipy Savitzky-Golay filter
- _binsize_ (int, allows to perform data binning to improve Signal-to-Noise)
- _legend_items_ (dict={scan_number:"name"}, overwrites generic legend names; works for the _load_ method)
- _legend_item_ (str, overwrites generic legend name in the _add_/_subtract_ method)


### 2d Images

Note: Can only load one scan at a time!

#### General loader for MCA detector data

```
load2d = Load2d()
load2d.load(config,'Filename.h5','x_stream','detector',1)
load2d.plot()
load2d.exporter()
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **x_stream** quantities include:
- All quantities contained in the sca folder(s) specified in the config
- All SCA specified in the config

4. Options for **detector** quantities include:
- All MCA specified in the config
- All STACK specified in the config with applied ROI

5. Select scan to analyse (comma-separated)

7. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]
- _grid_x_ (Takes a list with three arguments to apply 1d interpolation gridding)
  e.g. grid_x = [Start Energy, Stop Energy, Delta]
- _norm_by_ (Normalizes to specified stream)


### 2d histogram

```
mesh = LoadMesh()
mesh.load(config,'Filename.h5','x_stream','y_stream','z_stream',24)
mesh.plot()
mesh.exporter()
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **x_stream** quantities include:
- All quantities contained in the sca folder(s) specified in the config
- All SCA specified in the config
- All MCA specified in the config with ROI specified
- All STACK specified in the config with two ROIs specified

4. Options for **y_stream** quantities include:
- All quantities contained in the sca folder(s) specified in the config
- All SCA specified in the config
- All MCA specified in the config with ROI specified
- All STACK specified in the config with two ROIs specified

5. Options for **z_stream** quantities include:
- All quantities contained in the sca folder(s) specified in the config
- All SCA specified in the config
- All MCA specified in the config with ROI specified
- All STACK specified in the config with two ROIs specified

6. Specify scan to analyse

7. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]


### 3d Images

Note: Can only load one scan at a time!

```
load3d = Load2d()
load3d.load(config,'Filename.h5','stack',1)
load3d.plot()
load3d.export()
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **stack** quantities include:
- All STACK specified in the config

4. Select scan to analyse

7. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]
- _grid_x_ (Takes a list with three arguments to apply 1d interpolation gridding)
  e.g. grid_x = [Start Energy, Stop Energy, Delta]
- _norm_by_ (Normalizes to specified stream)


### Meta Data

```
bl = LoadBeamline()
bl.load(config,'Filename.h5','path to variable')
bl.plot()
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **path to variable** quantities include:
- All directory paths within the specified h5 file

### Spreadsheet

```
df = getSpreadsheet(config,'Filename.h5',columns=None)
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **columns** quantities include:
- Custom dictionary with column headers and quantities, see example below:

```
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
#columns['Comment'] = 'command'
columns['Status'] = 'status'
```