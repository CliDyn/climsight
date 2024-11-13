import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

class DataContainer:
    """
    A container class for managing data, figures, and datasets.

    Attributes:
    -----------
    df   : dict
        A dictionary to store pandas DataFrame to store tabular data.
    figs : dict
        A dictionary to store figures. example figs['haz_fig'] = {'fig':haz_fig,'source':source}
    data : dict
        A dictionary containing 'hist' and 'future' keys, each associated with an xarray Dataset.

    Methods:
    --------
    df:
        Property to get or set the DataFrame dictionary. Raises ValueError if the input is not a dictionary.
    figs:
        Property to get or set the figures dictionary. Raises ValueError if the input is not a dictionary.
    data:
        Property to get or set the data dictionary. Raises ValueError if the input is not a dictionary containing 'hist' and 'future' keys with xarray Datasets.
    """
    def __init__(self):
        self._df =   {}                  # Initialize as an empty dictionary for DataFrames
        self._figs = {}                  # Initialize as an empty dictionary for figures
        self._data = {}
        #{
        #    'hist': xr.Dataset(),        # Default to an empty xarray Dataset for historical data
        #    'future': xr.Dataset()       # Default to an empty xarray Dataset for future data
        #}

    @property
    def df(self):
        """Property to access the dictionary containing DataFrames."""
        return self._df

    @df.setter
    def df(self, value):
        """Set the DataFrame, must be a pandas DataFrame."""
        if isinstance(value, dict):
            self._df = value
        else:
            raise ValueError("df must be a dictionary of pandas DataFrame")

    @property
    def figs(self):
        """Property to access the figures dictionary."""
        return self._figs

    @figs.setter
    def figs(self, value):
        """Set the figures dictionary."""
        if isinstance(value, dict):
            self._figs = value
        else:
            raise ValueError("figs must be a dictionary")

    @property
    def data(self):
        """Property to access the data dictionary containing hist and future datasets."""
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, dict):
            self._figs = value
        else:
            raise ValueError("data must be a dictionary")        
        #"""Set the data dictionary, must be a dictionary with 'hist' and 'future' as xarray Datasets."""
        #if isinstance(value, dict) and 'hist' in value and 'future' in value:
        #    if isinstance(value['hist'], xr.Dataset) and isinstance(value['future'], xr.Dataset):
        #        self._data = value
        #    else:
        #        raise ValueError("Both 'hist' and 'future' must be xarray Datasets")
        #else:
        #    raise ValueError("data must be a dictionary containing 'hist' and 'future' keys")
