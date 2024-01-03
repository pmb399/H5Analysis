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

Before you start, you will need to import the required reixs package, enable bokeh plotting, and set the base directory.

```
## Setup necessarry inputs
from h5analysis.LoadData import *
from bokeh.io import show, output_notebook
output_notebook(hide_banner=True)
```

### 1d plots

#### General Loader1d

```
sca = Load1d()
sca.load('FileName.h5','x_stream','y_stream',1,2,3,4)  # Loads multiple scans individually
sca.add('FileName.h5','x_stream','y_stream',1,2,3,4)  # Adds multiple scans
sca.subtract('FileName.h5','x_stream','y_stream',1,2,3,4,norm=False) # Subtracts scans from the first scan
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
- All quantities in the header file
- _Mono Energy_ for the excitation energy
- _MCP Energy_ (uncalibrated)
- _SDD Energy_ (uncalibrated)
- _XEOL Energy_ (uncalibrated, actually the wavelength scale)
- _Points_ (by index)

4. Options for **y_stream** quantities include:
- All quantities in the header file
- _TEY_ (Total Electron Yield: sample normalized by mesh)
- _PFY_ (Partial Fluorescence Yield, normalized by mesh)
  Specify ROI with brackets:
  e.g. _PFY[490:560]_ for PFY from 490eV to 560eV
- _specPFY_ (spectrometer PFY, normalized by mesh)
  specify energy range
  e.g. specPFY[500:520]
- _XES_ (X-Ray emission and resonant x-ray emission at selected energies from the spectrometer MCP data)
  e.g. XES[560:565]
- _XRF_ (X-Ray fluorescence and resonant x-ray fluorescence at selected energies from the SDD data)
  e.g. XRF[550:570]
- _XEOL_ and (XEOL data from the optical spectrometer)
- _POY_ (Partial optical yield, normalized by mesh)
  e.g. POY[300:750]
- _Sample_ (Sample current, not normalized by mesh)
- _Mesh_ (Mesh current)

5. List all scans to analyse (comma-separated)

6. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]
- _background_ (Subtracts a XEOL background from XEOL scans)
  Set to True, uses the getXEOLback function with the background data stored (only supported with HDF5)
  Specify scan number, subtracts the XEOL scan taken at this particular scan
- _energyloss_ (Transfers the resultant MCP scale to energy loss 
  Set to True, then takes mean of mono energy array
  Specify float with the incident photon energy
- _grid_x_ (Takes a list with three arguments to apply 1d interpolation gridding)
  e.g. grid_x = [Start Energy, Stop Energy, Delta]
- _savgol_ (Takes a list with two or three arguments to apply data smoothing and derivatives)
  e.g. savgol = [Window length, Polynomial order, deriavtive] as specified in the scipy Savitzky-Golay filter
- _binsize_ (int, allows to perform data binning to improve Signal-to-Noise)
- _legend_items_ (dict={scan_number:"name"}, overwrites generic legend names; works for the _load_ method)
- _legend_item_ (str, overwrites generic legend name in the _add_/_subtract_ method)


### 2d Images

#### General loader for MCA detector data

Note: Can only load one scan at a time!

```
load2d = Load2d()
load2d.load('Filename.h5','x_stream','detector',1)
load2d.plot()
load2d.exporter()
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **x_stream** quantities include:
- All quantities in the header file
- _Mono Energy_ for the excitation energy

4. Options for **detector** quantities include:
- _SDD_ (SDD detector MCA)
- _MCP_ (MCP detector MCA)
- _XEOL_ (XEOL optical spectrometer MCA)

5. Select scan to analyse (comma-separated)

7. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]
- _background_ (Subtracts a XEOL background from XEOL scans)
  Set to True, uses the getXEOLback function with the background data stored (only supported with HDF5)
  Specify scan number, subtracts the XEOL scan taken at this particular scan
- _energyloss_ (Transfers the excitation-emission map to energy loss scale
- _grid_x_ (Takes a list with three arguments to apply 1d interpolation gridding)
  e.g. grid_x = [Start Energy, Stop Energy, Delta]


### 2d histogram

```
mesh = LoadMesh()
mesh.load('Filename.h5','x_stream','y_stream','z_stream',24)
mesh.plot()
mesh.exporter()
```

1. Create "Loader" object

2. Enter the file name of the scan to analyse ('FileName.h5')

3. Options for **x_stream** quantities include:
- All quantities in the header file
- _Mono Energy_ for the excitation energy
- _SDD Energy_ (Energy scale of the SDD detector)
- _MCP Energy_ (Energy scale of the MCP detector)
- _XEOL Energy_ (Wavelength scale of the XEOL optical spectrometer)

4. Options for **y_stream** quantities include:
- All quantities in the header file
- _Mono Energy_ for the excitation energy
- _SDD Energy_ (Energy scale of the SDD detector)
- _MCP Energy_ (Energy scale of the MCP detector)
- _XEOL Energy_ (Wavelength scale of the XEOL optical spectrometer)

5. Options for **z_stream** quantities include:
- All quantities in the header file
- All special quantities as specified for the Load1d() function

6. List all scans to analyse (comma-separated)

7. Set optional flags. Options include:
- _norm_ (Normalizes to [0,1])
- _xcoffset_ (Defines a constant shift in the x-stream)
- _xoffset_ (Takes a list of tuples and defines a polynomial fit of the x-stream)
- _ycoffset_ (Defines a constant shift in the y-stream)
- _yoffset_ (Takes a list of tuples and defines a polynomial fit of the y-stream)
  e.g. offset = [(100,102),(110,112),(120,121)]
- _background_ (Subtracts a XEOL background from XEOL scans)
  Set to True, uses the getXEOLback function with the background data stored (only supported with HDF5)
  Specify scan number, subtracts the XEOL scan taken at this particular scan
