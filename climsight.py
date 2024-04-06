import streamlit as st
import xarray as xr
import yaml
import numpy as np
from geopy.geocoders import Nominatim
import geopandas as gpd
from pyproj import Geod
import requests
import json
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import pandas as pd
from streamlit_folium import st_folium
import folium
import os
from langchain.callbacks.base import BaseCallbackHandler
from requests.exceptions import Timeout
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import yaml
import time
from shapely.geometry import Point, Polygon, LineString

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

model_name = config['model_name']
data_path = config['data_path']
coastline_shapefile = config['coastline_shapefile']
haz_path = config['haz_path']
pop_path = config['pop_path']
distance_from_event = config['distance_from_event']
lat_default = config['lat_default']
lon_default = config['lon_default']
year_step = config['year_step']
start_year = config['start_year']
end_year = config['end_year']
system_role = config['system_role']

clicked_coords = None
api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment

content_message = "{user_message} \n \
      Location: latitude = {lat}, longitude = {lon} \
      Adress: {location_str} \
      Where is this point?: {water_body_status} \
      Policy: {policy} \
      Additional location information: {add_properties} \
      Distance to the closest coastline: {distance_to_coastline} \
      Elevation above sea level: {elevation} \
      Current landuse: {current_land_use} \
      Current soil type: {soil} \
      Occuring species: {biodiv} \
      Current mean monthly temperature for each month: {hist_temp_str} \
      Future monthly temperatures for each month at the location: {future_temp_str}\
      Current precipitation flux (mm/month): {hist_pr_str} \
      Future precipitation flux (mm/month): {future_pr_str} \
      Current u wind component (in m/s): {hist_uas_str} \
      Future u wind component (in m/s): {future_uas_str} \
      Current v wind component (in m/s): {hist_vas_str} \
      Future v wind component (in m/s): {future_vas_str} \
      Natural hazards: {nat_hazards} \
      Population data: {population} \
      "


class StreamHandler(BaseCallbackHandler):
    """
    Taken from here: https://discuss.streamlit.io/t/langchain-stream/43782
    """

    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

@st.cache_data
def where_is_point(lat, lon):
    """
    Checks if a given point (latitude and longitude) is on land or water, and identifies the water body's name if applicable.


    Args:
    lat (float): Latitude of the point.
    lon (float): Longitude of the point.
    land_shapefile_path (str): Path to the land shapefile.

    Returns:
    Tuple: Indicates if the point is on land, in a lake, near a river, and the name of the lake or river if applicable.
    """

    point = Point(lon, lat)

    land_shp = gpd.read_file('./data/natural_earth/land/ne_10m_land.shp')
    is_on_land = land_shp.contains(point).any()
    print(f"Is the point on land? {'Yes.' if is_on_land else 'No.'}")
    if not is_on_land: # If point is not on land, no need to check for lakes or rivers
        water_body_status = "The selected point is in the ocean."
        return is_on_land, False, None, False, None, water_body_status

    water_shp_files = [
    './data/natural_earth/rivers/ne_10m_rivers_lake_centerlines.shp',
    './data/natural_earth/lakes/ne_10m_lakes.shp',
    './data/natural_earth/lakes/ne_10m_lakes_australia.shp',
    './data/natural_earth/lakes/ne_10m_lakes_europe.shp',
    './data/natural_earth/lakes/ne_10m_lakes_north_america.shp',
    './data/natural_earth/rivers/ne_10m_rivers_lake_centerlines.shp',
    './data/natural_earth/rivers/ne_10m_rivers_australia.shp',
    './data/natural_earth/rivers/ne_10m_rivers_europe.shp',
    './data/natural_earth/rivers/ne_10m_rivers_north_america.shp',
    ]
    water_shp = gpd.GeoDataFrame(pd.concat([gpd.read_file(shp) for shp in water_shp_files], ignore_index=True))

    
    # Initialize flags to check if the point is in water
    in_lake = False
    near_river = False
    lake_name = None 
    river_name = None

    for index, feature in water_shp.iterrows():
        geometry = feature.geometry

        if isinstance(geometry, Polygon) and geometry.contains(point):
            in_lake = True
            if 'name' in feature and pd.notnull(feature['name']):
                lake_name = feature.get('name')
            else:
                lake_name = None 
            break  # Stop the loop if the point is in a lake

        # create a buffer (in degree) around the river (line) and check if the point is within it
        elif isinstance(geometry, LineString) and geometry.buffer(0.005).contains(point):
            near_river = True
            if 'name' in feature and pd.notnull(feature['name']):
                river_name = feature.get('name')
            else:
                river_name = None 

    print(f"Is the point in a lake? {'Yes.' if in_lake else 'No.'} {'Name: ' + lake_name if lake_name else ''}")
    print(f"Is the point near a river? {'Yes.' if near_river else 'No.'} {'Name: ' + river_name if river_name else ''}")

    water_body_status = ""  # Initialize an empty string to store the status

    if is_on_land:
        water_body_status = "The selected point is on land"
        if in_lake:
            water_body_status = water_body_status + f" and in {'lake ' + lake_name if lake_name else 'a lake'}."
        elif near_river:
            water_body_status = water_body_status + f" and is near {'river ' + river_name if river_name else 'a river'}."
        else:
            water_body_status = water_body_status + "."

    return is_on_land, in_lake, lake_name, near_river, river_name, water_body_status

@st.cache_data
def get_location(lat, lon):
    """
    Returns the address of a given latitude and longitude using the Nominatim geocoding service.

    Parameters:
    lat (float): The latitude of the location.
    lon (float): The longitude of the location.

    Returns:
    dict: A dictionary containing information about the location.
    """
    # URL for Nominatim API reverse geocoding endpoint
    url = "https://nominatim.openstreetmap.org/reverse"

    params = {
        "lat": lat,
        "lon": lon,
        "format": "geojson",
        "extratags": 1,
        "namedetails": 1,
        "zoom": 18
    }
    headers = {
        "User-Agent": "climsight",
        "accept-language": "en"
    }
    response = requests.get(url, params=params, headers=headers, timeout=3)
    location = response.json()

    # Wait before making the next request (according to terms of use)
    # time.sleep(1)  # Sleep for 1 second

    if response.status_code == 200:
        location = response.json()        
        return location
    else:
        print("Error:", response.status_code, response.reason)
        return None


@st.cache_data
def get_adress_string(location):
    """
    Returns a tuple containing three strings:
    1. A string representation of the location address with all the key-value pairs in the location dictionary.
    2. A string representation of the location address with only the country, state, city and road keys in the location dictionary.
    3. Returning the country within in which the location has been clicked by user as a string.

    Parameters:
    location (dict): A dictionary containing the location address information.

    Returns:
    tuple: A tuple containing three strings.
    """
    address = location['features'][0]['properties']['address']

    location_str = "Address: "
    for key, value in address.items():
        location_str += f"{key}: {value}, "

    location_str = location_str.rstrip(', ') # remove the trailing comma and space

    location_str_for_print = "**Address:** "
    if "country" in address:
        location_str_for_print += f"{address['country']}, "
    if "state" in address:
        location_str_for_print += f"{address['state']}, "
    if "city" in address:
        location_str_for_print += f"{address['city']}, "
    if "road" in address:
        location_str_for_print += f"{address['road']} "
    if "house_number" in address:
        location_str_for_print += f"{address['house_number']}"

    location_str_for_print = location_str_for_print.rstrip(', ')
    country = address.get("country", "")

    return location_str, location_str_for_print, country

@st.cache_data
def get_location_details(location):
    """"
    Returns a dictionary containing:
    1. osm_type (e.g. way, node, relation)
    2. category (e.g. amenity, leisure, man_made, railway)
    3. type (e.g. outdoor_seating, hunting_stand, groyne, platform, nature_reserve)
    4. extratags (any additional information that is available, e.g. wikipeida links or opening hours)
    # 5. namedetails (full list of names, may include language variants, older names, references and brands) - currently excluded

    Parameters:
    location (dict): A dictionary containing the location address information.

    Returns:
    extracted_properties: A dictionary containing five elements.
    """
    properties = location['features'][0]['properties']
    extracted_properties = {key: properties[key] for key in ['osm_type', 'category', 'type', 'extratags']} #, 'namedetails']}

    return extracted_properties

@st.cache_data
def closest_shore_distance(lat: float, lon: float, coastline_shapefile: str) -> float:
    """
    Calculates the closest distance between a given point (lat, lon) and the nearest point on the coastline.

    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        coastline_shapefile (str): Path to the shapefile containing the coastline data

    Returns:
        float: The closest distance between the point and the coastline, in meters.
    """
    geod = Geod(ellps="WGS84")
    min_distance = float("inf")

    coastlines = gpd.read_file(coastline_shapefile)

    for _, row in coastlines.iterrows():
        geom = row["geometry"]
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                for coastal_point in line.coords:
                    _, _, distance = geod.inv(
                        lon, lat, coastal_point[0], coastal_point[1]
                    )
                    min_distance = min(min_distance, distance)
        else:  # Assuming LineString
            for coastal_point in geom.coords:
                _, _, distance = geod.inv(lon, lat, coastal_point[0], coastal_point[1])
                min_distance = min(min_distance, distance)

    return min_distance


@st.cache_data
def get_elevation_from_api(lat, lon):
    """
    Get the elevation of a location using the Open Topo Data API.

    Parameters:
    lat (float): The latitude of the location.
    lon (float): The longitude of the location.

    Returns:
    float: The elevation of the location in meters.
    """
    url = f"https://api.opentopodata.org/v1/etopo1?locations={lat},{lon}"
    response = requests.get(url, timeout=3)
    data = response.json()
    return data["results"][0]["elevation"]


@st.cache_data
def fetch_land_use(lon, lat):
    """
    Fetches land use data for a given longitude and latitude using the Overpass API.

    Args:
    - lon (float): The longitude of the location to fetch land use data for.
    - lat (float): The latitude of the location to fetch land use data for.

    Returns:
    - data (dict): A dictionary containing the land use data for the specified location.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    is_in({lat},{lon})->.a;
    area.a["landuse"];
    out tags;
    """
    response = requests.get(overpass_url, params={"data": overpass_query}, timeout=3)
    data = response.json()
    return data


@st.cache_data
def get_soil_from_api(lat, lon):
    """
    Retrieves the soil type at a given latitude and longitude using the ISRIC SoilGrids API.

    Parameters:
    lat (float): The latitude of the location.
    lon (float): The longitude of the location.

    Returns:
    str: The name of the World Reference Base (WRB) soil class at the given location.
    """
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=5"
        response = requests.get(url, timeout=3)  # Set timeout to 2 seconds
        data = response.json()
        return data["wrb_class_name"]
    except Timeout:
        return "not found"
    

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
def load_data():
    hist = xr.open_mfdataset(f"{data_path}/AWI_CM_mm_historical*.nc", compat="override")
    future = xr.open_mfdataset(f"{data_path}/AWI_CM_mm_ssp585*.nc", compat="override")
    return hist, future


def convert_to_mm_per_month(monthly_precip_kg_m2_s1):
    days_in_months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return monthly_precip_kg_m2_s1 * 60 * 60 * 24 * days_in_months


@st.cache_data
def extract_climate_data(lat, lon, _hist, _future):
    """
    Extracts climate data for a given latitude and longitude from historical and future datasets.

    Args:
    - lat (float): Latitude of the location to extract data for.
    - lon (float): Longitude of the location to extract data for.
    - _hist (xarray.Dataset): Historical climate dataset.
    - _future (xarray.Dataset): Future climate dataset.

    Returns:
    - df (pandas.DataFrame): DataFrame containing present day and future temperature, precipitation, and wind speed data for each month of the year.
    - data_dict (dict): Dictionary containing string representations of the extracted climate data.
    """
    hist_temp = hist.sel(lat=lat, lon=lon, method="nearest")["tas"].values - 273.15
    hist_temp_str = np.array2string(hist_temp.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    hist_pr = hist.sel(lat=lat, lon=lon, method="nearest")["pr"].values
    hist_pr = convert_to_mm_per_month(hist_pr)

    hist_pr_str = np.array2string(hist_pr.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    hist_uas = hist.sel(lat=lat, lon=lon, method="nearest")["uas"].values
    hist_uas_str = np.array2string(hist_uas.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    hist_vas = hist.sel(lat=lat, lon=lon, method="nearest")["vas"].values
    hist_vas_str = np.array2string(hist_vas.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    future_temp = future.sel(lat=lat, lon=lon, method="nearest")["tas"].values - 273.15
    future_temp_str = np.array2string(
        future_temp.ravel(), precision=3, max_line_width=100
    )[1:-1]

    future_pr = future.sel(lat=lat, lon=lon, method="nearest")["pr"].values
    future_pr = convert_to_mm_per_month(future_pr)
    future_pr_str = np.array2string(future_pr.ravel(), precision=3, max_line_width=100)[
        1:-1
    ]

    future_uas = future.sel(lat=lat, lon=lon, method="nearest")["uas"].values
    future_uas_str = np.array2string(
        future_uas.ravel(), precision=3, max_line_width=100
    )[1:-1]

    future_vas = future.sel(lat=lat, lon=lon, method="nearest")["vas"].values
    future_vas_str = np.array2string(
        future_vas.ravel(), precision=3, max_line_width=100
    )[1:-1]
    df = pd.DataFrame(
        {
            "Present Day Temperature": hist_temp[0, 0, :],
            "Future Temperature": future_temp[0, 0, :],
            "Present Day Precipitation": hist_pr[0, 0, :],
            "Future Precipitation": future_pr[0, 0, :],
            "Present Day Wind Speed": np.hypot(hist_uas[0, 0, :], hist_vas[0, 0, :]),
            "Future Wind Speed": np.hypot(future_uas[0, 0, :], future_vas[0, 0, :]),
            "Month": range(1, 13),
        }
    )
    data_dict = {
        "hist_temp": hist_temp_str,
        "hist_pr": hist_pr_str,
        "hist_uas": hist_uas_str,
        "hist_vas": hist_vas_str,
        "future_temp": future_temp_str,
        "future_pr": future_pr_str,
        "future_uas": future_uas_str,
        "future_vas": future_vas_str,
    }
    return df, data_dict


hist, future = load_data()

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

def get_population(pop_path, country):
    """
    Extracts population data (by UN) for a given country.

    Args:
    - pop_path: Path where the population data is stored
    - country: Takes the country which is returned by the geolocator

    Returns:
    - red_pop_data (pandas.DataFrame): reduced DataFrame containing present day and future values for only the following variables:
        - TPopulation1July (as of 1 July, thousands)
        - PopDensity (as of 1 July, persons per square km)
        - PopGrowthRate (percentage)
        - LEx (Life Expactancy at Birth, both sexes, in years)
        - NetMigrations (Number of Migrants, thousands)    
    """
    pop_dat = pd.read_csv(pop_path)

    unique_locations = pop_dat['Location'].unique()
    my_location = country

    # check if data is available for the country that we are currently investigating
    if my_location in unique_locations:
        country_data = pop_dat[pop_dat['Location'] == country]
        red_pop_data = country_data[['Time', 'TPopulation1July', 'PopDensity', 'PopGrowthRate', 'LEx', 'NetMigrations']]
        return red_pop_data
    else:
        print(f"No population data available {'for' +  country if country else ''}.")
        return None
    
def plot_population(pop_path, country):
    """
    Plots population data (by UN) for a given country.

    Args:
    - pop_path: Path where the population data is stored
    - country: Takes the country which is returned by the geolocator

    Returns:
    - plot: visual representation of the data distribution    
    """
    reduced_pop_data = get_population(pop_path, country)
    
    today = datetime.date.today()
    current_year = today.year

    if reduced_pop_data is not None and not reduced_pop_data.empty:
        fig, ax1 = plt.subplots(figsize=(10,6))
        plt.grid()

        # Total population data
        ax1.plot(reduced_pop_data['Time'], reduced_pop_data['TPopulation1July'], label='Total Population', color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('People in thousands', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # life expectancy
        ax2 = ax1.twinx()
        ax2.spines.right.set_position(('axes', 1.1))
        ax2.bar(reduced_pop_data['Time'], reduced_pop_data['LEx'], label='Life Expectancy', color='purple', alpha=0.1)
        ax2.set_ylabel('Life expectancy in years', color='purple', )
        ax2.tick_params(axis='y', labelcolor='purple')

        # population growth data
        ax3 = ax1.twinx()
        ax3.plot(reduced_pop_data['Time'], reduced_pop_data['PopGrowthRate'], label='Population Growth Rate', color='green')
        ax3.set_ylabel('Population growth rate in %', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Net Migrations
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(('axes', 1.2))
        ax4.plot(reduced_pop_data['Time'], reduced_pop_data['NetMigrations'], label='Net Migrations', color='black', linestyle='dotted')
        ax4.set_ylabel('Net migrations in thousands', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.axvline(x=current_year, color='orange', linestyle='--', label=current_year)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines+lines2+lines3+lines4, labels+labels2+labels3+labels4, loc='center right')

        plt.title(('Population changes in ' + country))
        return fig 
    else:
        return None
    
def calc_mean(years, dataset):
    """
    Calculates the mean of every column of a dataframe over a given time period and returns those means.

    Parameters:
    years (int): The time period that one is interested in to be averaged.
    dataset (pandas data frame): The corresponding data set. It has to have a column called 'Time' in datetime format.

    Returns:
    pandas data frame: A data frame with the means calculated for the given time span.
    """
    years = str(years) + 'Y'
    dataset.set_index('Time', inplace=True) # Set the 'Time' column as the index
    numeric_columns = dataset.select_dtypes(include='number')
    dataset = numeric_columns.resample(years).mean() # Resample the numeric data in x-year intervals and calculate the mean
    dataset.reset_index(inplace=True) # Reset the index to have 'Time' as a regular column
    dataset['Time'] = dataset['Time'].dt.year # and normal year format
    
    return dataset

def x_year_mean_population(pop_path, country, year_step=1, start_year=None, end_year=None):
    """
    Returns a reduced data set with the means calculated for every column over a given time span

    Parameters:
    pop_path (string): Path where the data is stored.
    country (string): The country which has been clicked on the map by the user.
    year_step (int): How many years shall be aggregated.
    start_year (int): The year from which onward population data is considered.
    end_year (int): The year until which population data is considered.

    Returns:
    pandas data frame: A data frame containing the mean population data values for a given time period.
    """
    # Check if start_year and end_year are within the allowed range
    if (start_year is not None and (start_year < 1950 or start_year > 2100)) or \
       (end_year is not None and (end_year < 1950 or end_year > 2100)):
        print("Warning: Start and end years must be between 1950 and 2100.")
        return None
    
    population_xY_mean = get_population(pop_path, country)
    if population_xY_mean is None:
        print(f"No population data available for {country}.")
        return None
    column_to_remove = ['LEx', 'NetMigrations'] # change here if less / other columns are wanted
    

    if not population_xY_mean.empty:
        population_xY_mean = population_xY_mean.drop(columns=column_to_remove)

        population_xY_mean['Time'] = pd.to_datetime(population_xY_mean['Time'], format='%Y')

        # Filter data based on start_year and end_year
        if start_year is not None:
            start_year = max(min(start_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year >= start_year]
        if end_year is not None:
            end_year = max(min(end_year, 2100), 1950)
            population_xY_mean = population_xY_mean[population_xY_mean['Time'].dt.year <= end_year]

        # Subdivide data into two data frames. One that contains the last complete x-year period (z-times the year_step) and the rest (modulo). For each data set the mean is calculated.
        modulo_years = len(population_xY_mean['Time']) % year_step 
        lastFullPeriodYear = population_xY_mean['Time'].dt.year.iloc[-1] - modulo_years  
        FullPeriod = population_xY_mean[population_xY_mean['Time'].dt.year <= lastFullPeriodYear]
        RestPeriod = population_xY_mean[population_xY_mean['Time'].dt.year > lastFullPeriodYear]

        # calculate mean for each period
        FullPeriodMean = calc_mean(year_step, FullPeriod)
        RestPeriodMean = calc_mean(modulo_years - 1, RestPeriod)
        RestPeriodMean = RestPeriodMean.iloc[1:] # drop first row as it will be same as last one of FullPeriodMean

        combinedMean  = pd.concat([FullPeriodMean, RestPeriodMean], ignore_index=True) # combine back into one data set

        new_column_names = {
            'TPopulation1July': 'TotalPopulationAsOf1July',
            'PopDensity': 'PopulationDensity',
            'PopGrowthRate': 'PopulationGrowthRate',
            # 'LEx': 'LifeExpectancy',
            # 'NetMigrations': 'NettoMigrations'  
        }
        combinedMean.rename(columns=new_column_names, inplace=True)

        return combinedMean
    
    else:
        return None

st.title(
    " :cyclone: \
         :ocean: :globe_with_meridians:  Climate Foresight"
)
# :umbrella_with_rain_drops: :earth_africa:  :tornado:

# Define map and handle map clicks
m = folium.Map(location=[lat_default, lon_default], zoom_start=13)
with st.sidebar:
    map_data = st_folium(m)
if map_data:
    clicked_coords = map_data["last_clicked"]
    if clicked_coords:
        lat_default = clicked_coords["lat"]
        lon_default = clicked_coords["lng"]

# Wrap the input fields and the submit button in a form
with st.form(key='my_form'):
    user_message = st.text_input(
        "Describe the activity that you would like to evaluate for this location:"
    )
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=lat_default, format="%.4f")
    lon = col2.number_input("Longitude", value=lon_default, format="%.4f")

    # Include the API key input within the form only if it's not found in the environment
    if not api_key:
        api_key_input = st.text_input(
            "OpenAI API key",
            placeholder="Enter your OpenAI API key here"
        )

    # Replace the st.button with st.form_submit_button
    submit_button = st.form_submit_button(label='Generate')


if submit_button and user_message:
    if not api_key:
        api_key = api_key_input
    if not api_key:
        st.error("Please provide an OpenAI API key.")
        st.stop()

    with st.spinner("Getting info on a point..."):

        st.markdown(f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}")

        is_on_land, in_lake, lake_name, near_river, river_name, water_body_status = where_is_point(lat, lon)
        if is_on_land:
            location = get_location(lat, lon)
            location_str, location_str_for_print, country = get_adress_string(location)
            add_properties = get_location_details(location)
            if not in_lake or not near_river:
                st.markdown(location_str_for_print)
            if in_lake:
                st.warning(f"You have clicked on {'lake ' + lake_name if lake_name else 'a lake'}. Our analyses are currently only meant for land areas. Please select another location for a better result.", icon="⚠️")
            if near_river:
                st.warning(f"You have clicked on a place that might be in {'the river ' + river_name if river_name else 'a river'}. Our analyses are currently only meant for land areas. Please select another location for a better result.", icon="⚠️")              
        else:
            st.warning("You have selected a place somewhere in the ocean. Our analyses are currently only meant for land areas. Please select another location for a better result.", icon="⚠️")
            country = None
            location_str = None
            add_properties = None
   
        try:
            elevation = get_elevation_from_api(lat, lon)
        except:
            elevation = "Not known"
        st.markdown(f"**Elevation:** {elevation} m")

        try:
            land_use_data = fetch_land_use(lon, lat)
        except:
            land_use_data = "Not known"
        try:
            current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
        except:
            current_land_use = "Not known"
        st.markdown(f"**Current land use:** {current_land_use}")

        try:
            soil = get_soil_from_api(lat, lon)
        except:
            soil = "Not known"
        st.markdown(f"**Soil type:** {soil}")

        biodiv = fetch_biodiversity(round(lat), round(lon))
        st.markdown(f"**Occuring species:** {biodiv}")

        distance_to_coastline = closest_shore_distance(lat, lon, coastline_shapefile)
        st.markdown(f"**Distance to the shore:** {round(distance_to_coastline, 2)} m")

        # create pandas dataframe
        df, data_dict = extract_climate_data(lat, lon, hist, future)
        # Plot the chart
        st.text(
            "Near surface temperature [source: AWI-CM-1-1-MR, historical, and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present Day Temperature", "Future Temperature"],
            color=["#d62728", "#2ca02c"],
        )
        st.text(
            "Precipitation [source: AWI-CM-1-1-MR, historical, and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present Day Precipitation", "Future Precipitation"],
            color=["#d62728", "#2ca02c"],
        )
        st.text(
            "Wind speed [source: AWI-CM-1-1-MR, historical, and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present Day Wind Speed", "Future Wind Speed"],
            color=["#d62728", "#2ca02c"],
        )

        filtered_events_square, promt_hazard_data = filter_events_within_square(lat, lon, haz_path, distance_from_event)

        population = x_year_mean_population(pop_path, country, year_step=year_step, start_year=start_year, end_year=end_year)

    policy = ""
    with st.spinner("Generating..."):
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box, display_method="write")
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            streaming=True,
            callbacks=[stream_handler],
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_role)
        human_message_prompt = HumanMessagePromptTemplate.from_template(content_message)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(
            llm=llm,
            prompt=chat_prompt,
            output_key="review",
            verbose=True,
        )

        output = chain.run(
            user_message=user_message,
            lat=str(lat),
            lon=str(lon),
            location_str=location_str,
            water_body_status=water_body_status,
            add_properties=add_properties,
            policy=policy,
            distance_to_coastline=str(distance_to_coastline),
            elevation=str(elevation),
            current_land_use=current_land_use,
            soil=soil,
            biodiv = biodiv,
            hist_temp_str=data_dict["hist_temp"],
            future_temp_str=data_dict["future_temp"],
            hist_pr_str=data_dict["hist_pr"],
            future_pr_str=data_dict["future_pr"],
            hist_uas_str=data_dict["hist_uas"],
            future_uas_str=data_dict["future_uas"],
            hist_vas_str=data_dict["hist_vas"],
            future_vas_str=data_dict["future_vas"],
            nat_hazards = promt_hazard_data,
            population = population,
            verbose=True,
        )

# Adding additional information after AI report
    haz_fig = plot_disaster_counts(filtered_events_square)
    population_plot = plot_population(pop_path, country)

    if haz_fig is not None or population_plot is not None:
        st.subheader("Additional information", divider='rainbow')

    # Natural Hazards
    if haz_fig is not None:
        st.markdown("**Natural hazards:**")
        st.pyplot(haz_fig)
        with st.expander("Source"):
            st.markdown('''
                *The GDIS data descriptor*  
                Rosvold, E.L., Buhaug, H. GDIS, a global dataset of geocoded disaster locations. Sci Data 8,
                61 (2021). https://doi.org/10.1038/s41597-021-00846-6  
                *The GDIS dataset*  
                Rosvold, E. and H. Buhaug. 2021. Geocoded disaster (GDIS) dataset. Palisades, NY: NASA
                Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/zz3b-8y61.
                Accessed DAY MONTH YEAR.  
                *The EM-DAT dataset*  
                Guha-Sapir, Debarati, Below, Regina, & Hoyois, Philippe (2014). EM-DAT: International
                disaster database. Centre for Research on the Epidemiology of Disasters (CRED).
            ''')

    # Population Data
    if population_plot is not None:
        st.markdown("**Population Data:**")
        st.pyplot(population_plot)
        with st.expander("Source"):
            st.markdown('''
            United Nations, Department of Economic and Social Affairs, Population Division (2022). World Population Prospects 2022, Online Edition. 
            Accessible at: https://population.un.org/wpp/Download/Standard/CSV/.
            ''')
    