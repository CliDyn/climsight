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
from functools import lru_cache
import os
import logging
import osmnx as ox


logger = logging.getLogger(__name__)

@lru_cache(maxsize=100)
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
    response = requests.get(url, params=params, headers=headers, timeout=10)
    location = response.json()

    # Wait before making the next request (according to terms of use)
    # time.sleep(1)  # Sleep for 1 second

    if response.status_code == 200:
        location = response.json()        
        return location
    else:
        print("Error:", response.status_code, response.reason)
        return None

@lru_cache(maxsize=100)
def is_point_onland(lat, lon, land_path_in):
    """
    Checks if a given point (latitude and longitude) is on land.

    Args:
    lat (float): Latitude of the point.
    lon (float): Longitude of the point.
    config (list): Config

    Returns:
    bool : True is on land, False is in the Ocean.
    str: water_status
    """
    point = Point(lon, lat)

    land_path = os.path.join(land_path_in, 'land/ne_10m_land.shp')
    land_shp = gpd.read_file(land_path)
    is_on_land = land_shp.contains(point).any()
    logging.info(f"Is the point on land? {'Yes.' if is_on_land else 'No.'}")
    
    water_body_status = ""  # Initialize an empty string to store the status
    if not is_on_land: # If point is not on land, no need to check for lakes or rivers
        water_body_status = "The selected point is in the ocean."
    
    return is_on_land, water_body_status

@lru_cache(maxsize=100)
def is_point_in_inlandwater(lat, lon, water_body_status="The selected point is on land."):
    """
    Checks if a given point (latitude and longitude) is in river or lake.

    Args:
    lat (float): Latitude of the point.
    lon (float): Longitude of the point.
    water_body_status (str): string if ocean/lake/...
        
    Returns:
    Tuple: Indicates if the point is on land, in a lake, near a river, and the name of the lake or river if applicable.
    """
    point = Point(lon, lat)

    #request to openstreetmap OSM with osmnx
    no_error = True
    try: 
        distance = 1000
        tags = {'natural':'water'}
        gdf = ox.features.features_from_point((lat,lon),tags,dist=distance)
    except Exception as e:
        logging.error(f"Unexpected error in request with osmnx ot OSM: {e}")
        #raise RuntimeError(f"Unexpected error in get_location: {e}")
        #if error, we assume point is on land
        is_inland_water = False
        no_error = False

    # Check if the point is within any of the geometries in the GeoDataFrame
    if no_error:
        contains_point = gdf['geometry'].apply(lambda geom: geom.contains(point))
        is_inland_water = contains_point.any()
        inland_water_name, inland_water_type = None, None
        if is_inland_water:
            gdf_contains_point = gdf[contains_point]
            try:
                inland_water_name = gdf_contains_point['name'].dropna().iloc[0]
            except:  
                inland_water_name = None
            try:
                inland_water_type = gdf_contains_point['water'].dropna().iloc[0]
            except:
                inland_water_type = None
            
            if inland_water_type:
                water_body_status = water_body_status.rstrip('.') + f" and located within the {inland_water_type if inland_water_type else ' water body'}."
            if inland_water_name:
                inland_water_name = water_body_status.rstrip('.') + f" named {'river ' + inland_water_name if inland_water_name else 'a river'}."

    return is_inland_water, water_body_status

@lru_cache(maxsize=100)
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


#@lru_cache(maxsize=100)
def get_adress_string(location):
    """
    Returns a tuple containing three strings:
    1. A string representation of the location address with all the key-value pairs in the location dictionary.
    2. A string representation of the location address with only the country, state, city and road keys in the location dictionary.
    3. Returning the country within in which the location has been clicked by user as a string.

    Parameters:
    location (dict): A dictionary containing the location address information.

    Returns:
    tuple: A tuple containing three strings, or (None, None, None) if address is unavailable.
    """
    # Check if location has the expected structure
    if not location or 'features' not in location:
        return None, None, None

    if not location['features'] or len(location['features']) == 0:
        return None, None, None

    if 'properties' not in location['features'][0] or 'address' not in location['features'][0]['properties']:
        return None, None, None

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


#@lru_cache(maxsize=100)
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
    extracted_properties: A dictionary containing properties, or empty dict if unavailable.
    """
    # Check if location has the expected structure
    if not location or 'features' not in location:
        return {}

    if not location['features'] or len(location['features']) == 0:
        return {}

    if 'properties' not in location['features'][0]:
        return {}

    properties = location['features'][0]['properties']
    # Extract only the keys that exist in properties
    extracted_properties = {key: properties.get(key, None) for key in ['osm_type', 'category', 'type', 'extratags']} #, 'namedetails']}

    return extracted_properties

@lru_cache(maxsize=100)
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


@lru_cache(maxsize=100)
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
    response = requests.get(url, timeout=10)
    data = response.json()
    return data["results"][0]["elevation"]

@lru_cache(maxsize=100)
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
    response = requests.get(overpass_url, params={"data": overpass_query}, timeout=10)
    data = response.json()
    return data

@lru_cache(maxsize=100)
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
        response = requests.get(url, timeout=10)  # Set timeout to 2 seconds
        data = response.json()
        return data["wrb_class_name"]
    except Timeout:
        return "not found"
  
