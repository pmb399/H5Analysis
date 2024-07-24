# Installation

## Prerequisites

To use the hdf5analysis package, the [python](https://www.python.org/) framework with version 3.9 or higher is required. For interactive computing, it is highly encouraged to make use of the interactive computing environment [Jupyter Notebook](https://github.com/jupyter).

::::{tab-set}

:::{tab-item} Anaconda3
A one-stop solution to obtaining python together with the jupyter notebook environment is the Anaconda3 distribution, available on the [Anadonda Website](https://www.anaconda.com/download). This python distribution also contains important scientific module that do not need to be installed manually, however, the convenience is paid with a lot of disk space.
:::

:::{tab-item} Manual Setup
You may proceed to install python for your operating system from the website above. The jupyter notebook environment may conveniently be installed via the pip package manager:

```
$ pip install notebook
```

Note, this only installs the minimum requirements to satifsy python and the jupyter notebook. Any other data analysis and scientific packages will need to be installed by hand.
:::

::::


## h5analysis package Installation

Install the package from PyPi with the pip package manager. This is the recommended way to obtain a copy for your local machine and will install all required package dependencies.

```
$ pip install h5analysis
```

If you wish to also be able to export 3d data stacks to movies, the FFMPEG python package and system binding is required. This is an optional dependency and the python integration can be installed together with h5analysis when triggering this optional dependency as shown below.

```
$ pip install h5analysis[FFMPEG]
```

Note, the export to MP4 file format using FFMPEG within python triggers the system's FFMPEG binding which may need to be installed. If you are using an Anaconda python environment, this task will be straight forward. In case you are using a different python eco-system, you will have to get the executable binary from the FFMPEG website.

::::{tab-set}

:::{tab-item} Anaconda3
Installing FFMPEG with conda is as simple as running 

```
$ conda install -c conda-forge ffmpeg
```

in your terminal.
:::

:::{tab-item} Manual Setup with FFMPEG Binary
Navigate to the [FFMPEG Website](https://ffmpeg.org/download.html). Download the executable file for your operating system and install FFMPEG. Ensure that the FFMPEG library is in your PATH.
:::

::::

To utilize the ```xraylarch``` engine for EXAFS data processing, ensure that the larch python package is installed:

```
$ pip install h5analysis[EXAFS]
```

To utilize the ```pybaselines``` package for background subtraction, ensure that the python package is properly installed:

```
$ pip install h5analysis[BASELINE]
```

## Activate widget extensions
In case that certain widgets aren't rendered properly, make sure to enable the appropriate jupyter extensions

```
$ jupyter nbextension enable --py widgetsnbextension
```