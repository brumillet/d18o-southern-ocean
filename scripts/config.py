# -*- coding: utf-8 -*-
"""
Paths and file names, in one place.

The notebooks used to open each file with a hard-coded name spelled out at the
call site, several of them more than once. Collecting the names here means a
renamed or moved file is a one-line change, and that a missing file produces a
clear error naming the variable to set rather than a bare FileNotFoundError
somewhere in the middle of a notebook.

The data directory is machine-specific. Set it either through the
``D18O_DATA_PATH`` environment variable or, from a notebook, with::

    import config
    config.set_data_path('D:/Data/d18o_so/')
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

# Where the input data lives. Override per machine, see the module docstring.
DATA_PATH = os.environ.get('D18O_DATA_PATH', 'D:/Data/d18o_so/')


# File names, relative to DATA_PATH.
FILES = {
    # d18O products
    'nemo_d18o': 'nemo_reconstructed_d18o.nc',
    'ocim_d18o': 'ocim_reconstructed_d18o.nc',
    'tmi_d18o': 'tmi_reconstructed_d18o.nc',
    'ls_climatology': 'WOCE_climatology_Lg&S_d18o.nc',
    'breitkreuz': 'D18O_Breitkreuz_et_al_2018.nc',

    # water-mass fractions
    'tmi_fractions': 'TMI_2deg_2010_water_mass_fractions.nc',
    'nemo_fractions': 'tm21ah21_extrapolated_dyes_regridded.nc',
    'ocim_fractions': 'ocim_steady_dyes.nc',
    'nemo_grid': 'basin_masks_orca1_nemo4p2.nc',

    # observations
    'giss': 'giss_d18o.txt',
    'giss_indian': 'giss_d18o_Indian.txt',
    'giss_modified': 'giss_d18o_modified.h5',
    'aoki': 'Datasheet_KY2018_ADS_20230718.csv',
    'locean': 'SI-Wisotopes-V3.csv',
    'bostock': 'd18o_helen_bostock.xlsx',
}


def set_data_path(path):
    """
    Point the project at a different data directory (call before loading).

    A trailing separator is added if missing. Forward slashes are used even on
    Windows, so that the path stays readable when it appears in a message.
    """
    global DATA_PATH
    DATA_PATH = path if path.endswith(('/', '\\')) else path + '/'
    return DATA_PATH


def data_file(key):
    """
    Absolute path of a known input file.

    Parameters
    ----------
    key : str
        One of the keys of :data:`FILES`.

    Returns
    -------
    str

    Raises
    ------
    KeyError
        If `key` is not a known file.
    FileNotFoundError
        If the file is not present, with the path and the variable to set.
    """
    if key not in FILES:
        raise KeyError(f'Unknown data file {key!r}; known keys: {sorted(FILES)}')
    path = DATA_PATH + FILES[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Set config.DATA_PATH (or the D18O_DATA_PATH '
            f'environment variable) to the directory holding {FILES[key]}.')
    return path


def figure_dir(name, create=True):
    """
    Path of a figure sub-directory, creating it on first use.

    Parameters
    ----------
    name : str
        Sub-directory of ``figures/``, e.g. ``'evaluation'``.
    create : bool, optional
        Create the directory if it does not exist (default True).

    Returns
    -------
    str
        Path ending with a separator, ready to be concatenated with a file name.
    """
    path = os.path.join(ROOT, 'figures', name) + os.sep
    if create:
        os.makedirs(path, exist_ok=True)
    return path
