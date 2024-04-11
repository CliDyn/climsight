""" 
Collection of functions for geographic data processing. 
These functions handle tasks such as location lookup, 
distance calculations, and geographic attributes 
extraction.
"""
import streamlit as st
import requests
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
import pandas as pd
from pyproj import Geod
from requests.exceptions import Timeout


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
  