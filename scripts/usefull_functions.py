# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:13:23 2024

@author: bmillet
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


# Reference the region based on the index of the 'fs_3d' field
dic_ocim = {'AA': 0, 'HS': 1, 'NP': 2, 'NA': 3, 'LLH': 4}
dyes = ['DyeLL', 'DyeMS',  'DyeNP', 'DyeHS', 'DyeNA', 'DyeAA']
dyes_TMI = ['dyeLL', 'dyeMS',  'dyeNP', 'dyeHS', 'dyeNA', 'dyeAA']
panels_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w']
panels_letter_parenthesis = ['(' + letter + ')' for letter in panels_letter]
models = ['TMI', 'OCIM', 'NEMO']

# Important information about the different datasets
coords_dataset = [(-40, -38), (-40, -55), (-60, -52), (-60, -78), (-40, -72), (0, -50), (20, -50), (35, -40), (50, -38), (50, -57), (80, -58), (-170, -50)]
l_neut_dens = [27.96, 28.01, 27.93, 28.18,
              28.14, 28.03, 28.07, 28.02,
              28.03, 28.05, 28.07, 27.99]
l_index = [0, 3, 4, 5, 7, 9, 10]
l_names_accepted = ['WOCE A11 (Meredith et al 1999)', 'GEOSECS Ostlund et al (1987)', "Drake's Passage (Meredith et al 1999)", 'Weppernig et al (1996)', 
                'Mackensen et al (1996)', 'Mackensen et al (1996)', 'ADOX1 (Frew et al 1995)', 'SWINDEX (K.J. Heywood pers. communication)', 
                'ADOX2 (Frew et al 1995)', 'ADOX1 (Frew et al 1995)', 'ADOX1 (Frew et al 1995)', 'GEOSECS Ostlund et al (1987)']
l_lon_accepted = [(-60, 20), (-70, 30), (-70, -10), (-60, -30),
                 (-60, -10), (-10, 30), (10, 40), (10, 60),
                  (50, 80), (50, 80), (80, 110), (160, 270)]


def create_l_depth():
    """
    Create a list of 45 depth levels for ocean depth discretization.
    
    The depth levels are structured as follows:
    - 0-50m: 10m intervals (6 levels)
    - 75-200m: 25m intervals (6 levels) 
    - 250-350m: 50m intervals (3 levels)
    - 400-1400m: 100m intervals (11 levels)
    - 1500-6000m: 250m intervals (19 levels)
    
    Returns:
    --------
    list
        List of 45 depth values in meters
    """
    l_depth0 = []
    for i in range(6):
        l_depth0.append(i*10)
    for i in range(6):
        l_depth0.append(75+i*25)
    for i in range(3):
        l_depth0.append(250+i*50)
    for i in range(11):
        l_depth0.append(400+i*100)
    for i in range(19):
        l_depth0.append(1500+i*250)
    return (l_depth0)



def approx_depth(d, l_depth):
    """
    Find the closest depth value in a depth list and return both value and index.
    
    Parameters:
    -----------
    d : float
        Target depth value to find the closest match for
    l_depth : list
        List of depth values to search in
        
    Returns:
    --------
    tuple
        (closest_depth_value, index_of_closest_value)
    """
    for i in range(0,len(l_depth)):
        diff = abs(d-l_depth[i])
        if i ==0:
            min = (diff,i)
        elif min[0] > diff:
            min = (diff,i)
    return((l_depth[min[1]],min[1]))


def create_dfs(aux):
    """
    Separate the d18O Southern Ocean compilation into 12 different datasets.
    
    Splits the data based on laboratory source and geographical location using
    predefined reference names and longitude boundaries. Handles longitude
    wrapping around the dateline (0°/360°).
    
    Parameters:
    -----------
    aux : pandas.DataFrame
        Input DataFrame containing d18O data with 'Reference' and 'Longitude' columns
        
    Returns:
    --------
    list of pandas.DataFrame
        List of 12 DataFrames, each containing data from a specific 
        laboratory/geographical region
    """
    dfs = []; l_names_dfs = []
    for i in range(len(l_names_accepted)):
        lon_min, lon_max, name = l_lon_accepted[i][0], l_lon_accepted[i][1], l_names_accepted[i]
        l_names_dfs.append(name)
        if (lon_min < 0) & (lon_max < 0):
            dfs.append(aux.where((aux['Reference'] == name) & (aux['Longitude'] >= 360 + lon_min) & (aux['Longitude'] <= 360 + lon_max)).dropna())
        elif (lon_min < 0):
            dfs.append(aux.where((aux['Reference'] == name) & ((aux['Longitude'] >= 360 + lon_min) | (aux['Longitude'] <= lon_max))).dropna())
        else:
            dfs.append(aux.where((aux['Reference'] == name) & (aux['Longitude'] >= lon_min) & (aux['Longitude'] <= lon_max)).dropna())
            
    return dfs
    

    
def get_BoundNorm(vmin, vmax, cmapname = 'coolwarm', nbins = 10):
    """
    Create a normalized colormap with discrete color levels for plotting.
    
    Parameters:
    -----------
    vmin : float
        Minimum value for the colormap range
    vmax : float
        Maximum value for the colormap range
    cmapname : str, optional
        Name of the matplotlib colormap (default: 'coolwarm')
    nbins : int, optional
        Number of discrete color bins (default: 10)
        
    Returns:
    --------
    matplotlib.colors.BoundaryNorm
        Normalized colormap object for use with matplotlib plotting functions
    """
    cmap = plt.colormaps[cmapname]
    levels = MaxNLocator(nbins = nbins).tick_values(vmin, vmax)
    return(BoundaryNorm(levels, ncolors = cmap.N, clip=True))
    


def get_xylabels(n_lines, n_col, i):
    """
    Determine whether to show x and y axis labels for subplot in a grid layout.
    
    Returns True for x-labels only on the bottom row and True for y-labels
    only on the leftmost column of a subplot grid.
    
    Parameters:
    -----------
    n_lines : int
        Total number of rows in the subplot grid
    n_col : int
        Total number of columns in the subplot grid
    i : int
        Linear index of the current subplot (0-based)
        
    Returns:
    --------
    tuple of bool
        (xlabel, ylabel) - True if labels should be shown, False otherwise
        
    Example:
    --------
    >>> get_xylabels(3, 2, 4)  # Bottom left subplot in 3x2 grid
    (True, True)
    """
    i_line, i_col = i//n_col, i%n_col
    xlabel, ylabel = False, False
    if i_line == (n_lines - 1):
        xlabel = True
    if i_col == 0:
        ylabel = True
    return(xlabel, ylabel)



def plot_details_axis(ax, pco, cb = True, cbarlabel = '', xlim = (-50, 60), ylim = (6000, 0), font = 15, cmapname = 'viridis', nbins = 10, title=''
                  , xlabels = True, ylabels = True, xticks = [-40+20*i for i in range(6)], yticks = [1000*i for i in range(7)]):
    """
    Configure axis properties, labels, and colorbar for oceanographic plots.
    
    This function standardizes the appearance of plots by setting axis limits,
    tick marks, labels, and optionally adding a colorbar. Designed specifically
    for depth vs. latitude oceanographic data visualization.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to configure
    pco : matplotlib.collections or similar
        Plot object (e.g., from pcolormesh, scatter) for colorbar creation
    cb : bool, optional
        Whether to add a colorbar (default: True)
    cbarlabel : str, optional
        Label for the colorbar (default: '')
    xlim : tuple, optional
        X-axis limits (default: (-50, 60))
    ylim : tuple, optional
        Y-axis limits (default: (6000, 0))
    font : int, optional
        Font size for labels and ticks (default: 15)
    cmapname : str, optional
        Colormap name (default: 'viridis')
    nbins : int, optional
        Number of colorbar bins (default: 10)
    title : str, optional
        Plot title (default: '')
    xlabels : bool, optional
        Whether to show x-axis labels (default: True)
    ylabels : bool, optional
        Whether to show y-axis labels (default: True)
    xticks : list, optional
        X-axis tick positions (default: latitude ticks)
    yticks : list, optional
        Y-axis tick positions (default: depth ticks in km)
        
    Returns:
    --------
    None
        Modifies the axes object in place
    """
    if cb:
        cbar = plt.colorbar(pco)#, ticks = [i*age_max/4 for i in range(6)])
        cbar.ax.tick_params(labelsize=font-2)
        cbar.set_label(cbarlabel, fontsize = font)
        
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if xlabels:
        if xticks == [-40+20*i for i in range(6)]:
            ax.set_xlabel('Latitude (°N)', fontsize=font)
        elif xticks == [-80 + i * 30 for i in range(5)]:
            ax.set_xlabel('Latitude (°N)', fontsize=font)
    else: 
        ax.set_xticklabels([])
    if ylabels:
        if yticks == [1000*i for i in range(7)]:
            ax.set_yticklabels([1*i for i in range(7)],fontsize = font)
            ax.set_ylabel('Depth (km)', fontsize=font)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize = font)
    ax.set_title(title, fontsize = font)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    


def add_cbar(fig, pco, x = 0.92, y = 0.1, width = 0.015, height = 0.8, fontsize = 14, label = '', ticks = []):
    """
    Add a custom positioned colorbar to a matplotlib figure.
    
    Creates a colorbar at a specified position with customizable dimensions,
    labels, and tick marks. Useful for complex subplot layouts where automatic
    colorbar positioning is insufficient.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to add the colorbar to
    pco : matplotlib.collections or similar
        Plot object (e.g., from pcolormesh, contourf) for colorbar creation
    x : float, optional
        Left edge position in figure coordinates (0-1) (default: 0.92)
    y : float, optional
        Bottom edge position in figure coordinates (0-1) (default: 0.1)
    width : float, optional
        Width in figure coordinates (0-1) (default: 0.015)
    height : float, optional
        Height in figure coordinates (0-1) (default: 0.8)
    fontsize : int, optional
        Font size for labels and ticks (default: 14)
    label : str, optional
        Colorbar label text (default: '')
    ticks : list, optional
        Custom tick positions. If empty, uses default ticks (default: [])
        
    Returns:
    --------
    matplotlib.colorbar.Colorbar
        The created colorbar object
    """
    cax = fig.add_axes([x, y, width, height])  # [x, y, width, height]
    cb = plt.colorbar(pco, cax=cax)
    cb.ax.tick_params(labelsize=fontsize - 1)
    cb.set_label(label, fontsize=fontsize)
    if ticks != []:
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=fontsize)
    return cb



def non_monotone(l):
    """
    Find the first value where increasing monotonicity breaks in a sequence.
    
    Designed specifically for longitude arrays to detect where values decrease,
    indicating a potential dateline crossing that needs coordinate transformation.
    
    Parameters:
    -----------
    l : array-like
        Input sequence to check for monotonicity (NaN values are filtered out)
        
    Returns:
    --------
    tuple
        (value, index) where monotonicity breaks, or (-1, None) if sequence 
        is monotonic or empty
        
    Notes:
    ------
    Originally designed for nav_lon arrays in NEMO ocean model output to
    handle longitude coordinate wrapping around the dateline.
    """
    l = [x for x in l if not np.isnan(x)]
    if len(l) == 0:
        return((-1, None))
        
    aux = l[0]
    for i in range(len(l)):
        if l[i] < aux:
            return (l[i], i)
        aux = l[i]
            
    return((-1, None))



def transfo(nav_lat, nav_lon):
    """
    Transform NEMO navigation arrays for continuous longitude plotting.
    
    Handles longitude wrapping around the dateline by adding 360° to values
    after discontinuities. Also processes latitude arrays by replacing 
    missing values with appropriate fill values.
    
    Parameters:
    -----------
    nav_lat : numpy.ndarray
        2D array of latitude coordinates from NEMO grid
    nav_lon : numpy.ndarray  
        2D array of longitude coordinates from NEMO grid
        
    Returns:
    --------
    tuple of numpy.ndarray
        (transformed_nav_lat, transformed_nav_lon) ready for plotting
    """
    nav_lon = np.where(nav_lon != -1, nav_lon, np.nan)
    
    nav_lon2 = nav_lon.copy()
    for i in range(np.shape(nav_lat)[0]):
        value = non_monotone(nav_lon2[i])[0]
        if value != -1:
            index = list(nav_lon2[i]).index(value)
            nav_lon2[i][index:] += 360
        
    nav_lon2[np.isnan(nav_lon2)] = -1
    nav_lat[nav_lat == -1] = np.nan
    nav_lat[np.isnan(nav_lat)] = -100
    return(nav_lat, nav_lon2)



def get_new_nav_lat_lon(data_path):
    """
    Load and process NEMO grid coordinates from basin mask file.
    
    Reads navigation coordinates from the ORCA1 NEMO grid file and applies
    coordinate transformation for continuous plotting across the dateline.
    
    Parameters:
    -----------
    data_path : str
        Path to directory containing 'basin_masks_orca1_nemo4p2.nc' file
        
    Returns:
    --------
    tuple of numpy.ndarray
        (nav_lat, nav_lon) - Processed 2D coordinate arrays ready for plotting
    """
    new_grid = xr.open_dataset(data_path + 'basin_masks_orca1_nemo4p2.nc')
    new_nav_lat, new_nav_lon = new_grid['nav_lat'].values.copy(), new_grid['nav_lon'].values.copy()
    new_grid.close()
    return(transfo(new_nav_lat, new_nav_lon))


def index_to_exclude(dfs2, suffix):
    """
    Identify dataset indices to exclude based on water mass fraction criteria.
    
    Determines which datasets should be excluded from optimization based on
    insufficient representation (< 0.2) of key Southern Ocean water masses (Antarctic, 
    North Atlantic, and High Salinity water masses).
    
    Parameters:
    -----------
    dfs2 : list of pandas.DataFrame
        List of datasets containing water mass fraction data
    suffix : str
        Model suffix for column names (e.g., '_TMI', '_OCIM', '_NEMO')
        
    Returns:
    --------
    list of int
        Indices of datasets to exclude from analysis
    """
    index_to_exclude = []
    for j in range(len(dfs2)):
        if not (np.nanmax(dfs2[j]['dyeAA' + suffix]) >= 0.2) & (np.nanmax(dfs2[j]['dyeNA' + suffix]) >= 0.2) & (np.nanmax(dfs2[j]['dyeHS' + suffix]) >= 0.2):
            index_to_exclude.append(j)
    index_to_exclude.append(11)
    return index_to_exclude


def index_to_exclude_dye(dfs2, dye, suffix):
    """
    Identify dataset indices to exclude for a specific water mass (dye).
    
    Determines which datasets should be excluded from optimization for a
    particular water mass based on insufficient fraction representation (< 0.2).
    
    Parameters:
    -----------
    dfs2 : list of pandas.DataFrame
        List of datasets containing water mass fraction data
    dye : str
        Water mass identifier (e.g., 'dyeAA', 'dyeNA', 'dyeHS', 'dyeMS')
    suffix : str
        Model suffix for column names (e.g., '_TMI', '_OCIM', '_NEMO')
        
    Returns:
    --------
    list of int
        Indices of datasets to exclude for this specific water mass
    """
    index_to_exclude_dye = []
    for j in range(len(dfs2)):
        if not (np.nanmax(dfs2[j][dye + suffix]) >= 0.2):
            index_to_exclude_dye.append(j)
    return index_to_exclude_dye


def rmse_model_recons(full_tracer_data, tracer_column, model, mean_end_members):
    """
    Calculate RMSE between observed and reconstructed tracer values using mean end-member values.
    
    Parameters:
    -----------
    full_tracer_data : pd.DataFrame
        DataFrame containing observed tracer data and water mass fractions
    tracer_column : str
        Name of the column containing observed tracer values (e.g., 'd18O')
    model : str
        Model name ('TMI', 'OCIM', 'NEMO')
    mean_end_members : dict
        Dictionary with mean end-member values from optimization
        
    Returns:
    --------
    float
        RMSE value between observations and reconstructions
    """
    obs_values = full_tracer_data[tracer_column].values
    recon_values = np.zeros_like(obs_values)
    
    # Calculate reconstructed values using mean end-member values
    for dye in dyes_TMI:
        column_name = f'{dye}_{model}'
        if column_name in full_tracer_data.columns:
            em_value = mean_end_members[tracer_column][model][dye]
            recon_values += full_tracer_data[column_name].values * em_value
        else:
            print(f"Warning: Column {column_name} not found in data.")
    
    # Calculate RMSE with valid data points only
    valid_mask = ~np.isnan(obs_values) & ~np.isnan(recon_values)
    if np.any(valid_mask):
        rmse_value = np.sqrt(np.mean((obs_values[valid_mask] - recon_values[valid_mask])**2))
    else:
        rmse_value = np.nan
        
    return rmse_value