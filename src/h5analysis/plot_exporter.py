"""Triggers the matplotlib engine for plot exports"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import colorcet


def create_figure(figsize=None, x_minor_ticks=None, x_major_ticks=None, y_minor_ticks=None, y_major_ticks=None, top=True, right=True, fontsize_axes='medium', fontsize_labels='medium', fontsize_title='medium', title_pad=8, xlabel="", ylabel="", title="", xlim=None, ylim=None, **kwargs):
    """
    Create a matplotlib plot window

    Parameters
    ----------

    figsize: tuple
        determines size of plot
    x_minor_ticks: float
        distance between minor ticks on primary axis
    x_major_ticks: float
        distance between major ticks on primary axis
    y_minor_ticks: float
        distance between minor ticks on secondary axis
    y_major_ticks: float
        distance between major ticks on secondary axis
    top: Boolean
        Display ticks on top of the plot
    right: Boolean
        Display ticks on the right of the plot
    fontsize_axes: string or int
        Set the fontsize of the axes ticks
    fontsize_labels: string or int
        Set fontsize of the axis labels
    fontsize_title: string or int
        Set fontsize of the title
    title_pad: int
        Padding between title and the top of the plot
    xlabel: string
        Label of the primary axis
    ylabel: string
        Label of the secondary axis
    title: string
        Title displayed at the top of the plot
    xlim: tuple
        Limits the visible x-range
    ylim: tuple
        Limits the visible y-range
    """

    # Set size of figure
    if figsize == None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))
    ax1 = plt.axes()

    # Set the ticks on both axes
    if x_minor_ticks != None:
        ax1.xaxis.set_minor_locator(MultipleLocator(x_minor_ticks))
    if x_major_ticks != None:
        ax1.xaxis.set_major_locator(MultipleLocator(x_major_ticks))
    if y_minor_ticks != None:
        ax1.yaxis.set_minor_locator(MultipleLocator(y_minor_ticks))
    if y_major_ticks != None:
        ax1.yaxis.set_major_locator(MultipleLocator(y_major_ticks))

    ax1.tick_params(which='major', length=7, top=top, right=right)
    ax1.tick_params(which='minor', length=4, width=1, top=top, right=right)

    for tick in ax1.xaxis.get_major_ticks():
        try:
            tick.label.set_fontsize(fontsize_axes)
        except:
            tick.label1.set_fontsize(fontsize_axes)
            tick.label2.set_fontsize(fontsize_axes)
    for tick in ax1.yaxis.get_major_ticks():
        try:
            tick.label.set_fontsize(fontsize_axes)
        except:
            tick.label1.set_fontsize(fontsize_axes)
            tick.label2.set_fontsize(fontsize_axes)

    # Set the labels
    if title != "":
        plt.title(title, fontsize=fontsize_title, pad=title_pad)
    plt.xlabel(xlabel, fontsize=fontsize_labels)
    plt.ylabel(ylabel, fontsize=fontsize_labels)

    # Set the visible range
    if xlim != None:
        plt.xlim(xlim[0], xlim[1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])

    return fig, ax1


def plot_image(fig, ax, x, y, z, cmap='CET_R4', levels=None, aspect='equal', colorbar=True, zlabel="", fontsize_colorbar='medium', **kwargs):
    """
    Plots a contourplot in the created figure

    Parameters
    ----------

    fig: object 
        matplotlib Figure object
    ax: object
        matplotlib Axes object
    x: 1d array
        coordinates of the primary axis
    y: 1d array
        coordinates of the secondary axis
    z: 1d array
        values corresponding to the (x,y) pairs
    cmap: string
        name of matplotlib colourmap
    levels: int
        determines how many levels the z data should be binned in
    aspect: [equal, auto]
        Set the axis to scale or stretch figszize
    colorbar: Boolean
        Display a colorbar
    zlabel: string
        Label of the colorbar
    fontsize_colorbar: string or int
        Fontsize of the colorbar ticks
    """

    # Get special colour mapper
    if cmap == 'CET_R4':
        cmap = cm.get_cmap('cet_CET_R4')

    # Create contourplot
    if levels != None:
        img = ax.contourf(x, y, z, cmap=cmap, levels=levels)
    else:
        img = ax.contourf(x, y, z, cmap=cmap)

    # Set aspect ratio of axes
    ax.set_aspect(aspect)

    # Plot a colorbar with label if requested
    if colorbar == True:
        cbar = fig.colorbar(img)
        if zlabel != "":
            cbar.ax.set_ylabel(
                zlabel, fontsize=ax.get_xaxis().label.get_font().get_size())
        cbar.ax.tick_params(labelsize=fontsize_colorbar)

    return fig


def plot_lines(fig, ax, data_list, legend=False, fontsize_legend=12, **kwargs):
    """
    Plots Line1d objects in the created figure

    Parameters
    ----------

    fig: object 
        matplotlib Figure object
    ax: object
        matplotlib Axes object
    data_list: list
        list of dictionaries containing the keys
        * x
        * y
        * yoffset
        * label
        * linewidth
    legend: Boolean
        Show/Hide plot legend
    fontsize_legend: int
        Fontsize of the legend entries
    """

    twin_y = False

    # Add all lines to plot
    for data in data_list:
        if data['twin_y'] == True and twin_y == False:
            twin_y = True
            ax2 = ax.twinx()

        plot_kwargs = data.copy()
        plot_kwargs.pop('x')
        plot_kwargs.pop('y')
        plot_kwargs.pop('twin_y')

        if data['twin_y'] == False:
            ax.plot(data['x'], data['y'],**plot_kwargs)
        else:
            ax2._get_lines = ax._get_lines
            ax2.plot(data['x'], data['y'],**plot_kwargs)

    # Generate legend if requested
    if legend == True:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1+handles2,labels1+labels2,fontsize=fontsize_legend)

    return fig


def save_figure(fig, fname, data_format='pdf', **kwargs):
    """
    Export displayed data as processed plot

    Parameters
    ----------

    fig: object 
        matplotlib figure object
    fname: string
        path and file name of the exported file
    data_format: string, [pdf,svg,png]
        Sets the output data format and matplotlib backend used
    """

    fig.savefig(f"{fname}.{data_format}", bbox_inches='tight')
