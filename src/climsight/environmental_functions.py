"""
Functions for enviromental data extracting, processing, analysis, plotting.
like biodiversity, natural hazards ....
"""
   
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

@st.cache_data
def fetch_biodiversity(lon, lat):
    """
    Fetches biodiversity data for a given longitude and latitude using the GBIF API.

    Args:
    - lon (float): The longitude of the location to fetch biodiversity data for.
    - lat (float): The latitude of the location to fetch biodiversity data for.

    Returns:
    - data (dict): A dictionary containing the biodiversity data for the specified location.
    """
    gbif_api_url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "decimalLatitude": lat,
        "decimalLongitude": lon,
    }
    response = requests.get(gbif_api_url, params=params, timeout=3)
    biodiv = response.json()
    biodiv_set = set()
    if biodiv['results']:
        for record in biodiv['results']:
            if 'genericName' in record and record.get('taxonRank') != 'UNRANKED':
                biodiv_set.add(record['genericName'])
        biodiversity = ', '.join(list(biodiv_set))
    else:
        biodiversity = "Not known"
    return biodiversity
 
 
@st.cache_data
def load_nat_haz_data(haz_path):
    """
    Load natural hazard data from a CSV file and filter relevant columns.

    Args:
    - haz_path (str): File path to the CSV file containing natural hazard data.

    Returns:
    - pandas.DataFrame: Dataset with selected columns ('country', 'year', 'geolocation', 'disastertype', 'latitude', 'longitude').
    """

    haz_dat = pd.read_csv(haz_path)

    # reduce data set to only contain relevant columns
    columns_to_keep = ['country', 'year', 'geolocation', 'disastertype', 'latitude', 'longitude']
    haz_dat = haz_dat.loc[:, columns_to_keep]

    return(haz_dat)

@st.cache_data
def filter_events_within_square(lat, lon, haz_path, distance_from_event):
    """
    Filter events within a square of given distance from the center point.

    Args:
    - lat (float): Latitude of the center point (rounded to 3 decimal places)
    - lon (float): Longitude of the center point (rounded to 3 decimal places)
    - haz_dat (pandas.DataFrame): Original dataset.
    - distance_from_event (float): Distance in kilometers to form a square.

    Returns:
    - pandas.DataFrame: Reduced dataset containing only events within the square.
    """

    haz_dat = load_nat_haz_data(haz_path)

    # Calculate the boundaries of the square
    lat_min, lat_max = lat - (distance_from_event / 111), lat + (distance_from_event / 111)
    lon_min, lon_max = lon - (distance_from_event / (111 * np.cos(np.radians(lat)))), lon + (distance_from_event / (111 * np.cos(np.radians(lat))))

    # Filter events within the square
    filtered_haz_dat = haz_dat[
        (haz_dat['latitude'] >= lat_min) & (haz_dat['latitude'] <= lat_max) &
        (haz_dat['longitude'] >= lon_min) & (haz_dat['longitude'] <= lon_max)
    ]

    prompt_haz_dat = filtered_haz_dat.drop(columns=['country', 'geolocation', 'latitude', 'longitude'])

    return filtered_haz_dat, prompt_haz_dat

@st.cache_data
def plot_disaster_counts(filtered_events):
    """
    Plot the number of different disaster types over a time period for the selected location (within 5km radius).

    Args:
    - filtered_events: Only those natural hazard events that were within a 5 km (or whatever other value is set for distance_from_event) radius of the clicked location.
    Returns:
    - figure: bar plot with results
    """
    if not filtered_events.empty:
        # Group by 'year' and 'disastertype' and count occurrences
        disaster_counts = filtered_events.groupby(['year', 'disastertype']).size().unstack(fill_value=0)
        place = filtered_events['geolocation'].unique()

        # create figure and axes
        fig, ax = plt.subplots(figsize=(10,6))
        
        # Plotting the bar chart
        disaster_counts.plot(kind='bar', stacked=False, ax=ax, figsize=(10,6), colormap='viridis')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Count of different disaster types in ' + place[0] + ' over time')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend(title='Disaster Type')

        return fig
    else:
        return None
