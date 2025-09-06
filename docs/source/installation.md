# Installation

## Prerequisites

To use the hdf5analysis package, the [python](https://www.python.org/) framework with version 3.9 or higher is required. For interactive computing, it is highly encouraged to make use of the interactive computing environment [Jupyter Notebook](https://github.com/jupyter).

::::{tab-set}

:::{tab-item} Anaconda3/Miniforge
A one-stop solution to obtaining python together with the jupyter notebook environment is the Anaconda3 distribution, available on the [Anadonda Website](https://www.anaconda.com/download). This python distribution also contains important scientific module that do not need to be installed manually, however, the convenience is paid with a lot of disk space. Alternatively, if your organization can not use Anacoda due to licence requirements Miniforge is an alternative, available on the [Minforge Website](https://conda-forge.org/miniforge).
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

You will need to ffmpeg binaries.
```
$ conda install ffmpeg
```

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