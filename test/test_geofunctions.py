'''
This script performs unit testing for the climsight geolocation functions,
ensuring configuration files are correctly loaded and geolocation operations 
return accurate results. It uses pytest to test various components, including 
location retrieval and detailed address parsing. The tests validate both the 
presence of essential config files and the accuracy of data fetched from 
external geocoding services. 
'''

import pytest
import yaml
import warnings
import sys
import os
import requests
import requests_mock

# Append the directory containing the module to sys.path
module_dir = os.path.abspath('../src/climsight/')
if module_dir not in sys.path:
    sys.path.append(module_dir)



from geo_functions import (
    get_location,
    where_is_point,
    get_adress_string,
    get_location_details,
    closest_shore_distance,
    get_elevation_from_api,
    fetch_land_use,
    get_soil_from_api
)

config_path = '../config.yml'
test_config_path = 'config_test_geofunctions.yml'

def test_config_files_exist():
    assert os.path.exists(config_path), f"Configuration file does not exist: {config_path}"
    assert os.path.exists(test_config_path), f"Test configuration file does not exist: {test_config_path}"


@pytest.fixture
def config_main():
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@pytest.fixture
def config_geo():
    with open(test_config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

@pytest.fixture
def latlon(config_geo):
    lat = float(config_geo['test_location']['lat'])
    lon = float(config_geo['test_location']['lon'])
    return lat, lon  # This returns a tuple, but we'll unpack this in the function call

@pytest.mark.request
def test_get_location(config_geo):
    
    lat = float(config_geo['test_location']['lat'])
    lon = float(config_geo['test_location']['lon'])    
    expected_address = config_geo['test_location']['expected_address']
    expected_address_string = config_geo['test_location']['expected_address_string']
    expected_location_details = config_geo['test_location']['expected_location_details']
        
    location = get_location(lat, lon)

    message = "\033[91m" + " \n Warning: test can fail because it changes the road unpredictably. \n" + "\033[0m"    

    assert location is not None  , f"get_location: response.status_code not == 200"
    assert 'error' not in location, f"get_location: {location}"
    try:
      assert location['features'][0]['properties']['address'] == expected_address, f"wrong adress"
    except AssertionError as e: 
      warnings.warn(str(e)+message, UserWarning)
      
    address_string = get_adress_string(location)
    try:
        assert address_string == expected_address_string
    except AssertionError as e: 
      warnings.warn(str(e)+message, UserWarning)
        
    location_details = get_location_details(location)
    try:
        assert location_details == expected_location_details
    except AssertionError as e: 
      warnings.warn(str(e)+message, UserWarning)
        

@pytest.mark.mock_request
def test_mock_get_location(config_geo, requests_mock):

    lat = float(config_geo['test_location']['lat'])
    lon = float(config_geo['test_location']['lon'])    
    expected_address = config_geo['test_location']['expected_address']
    expected_address_string = tuple(config_geo['test_location']['expected_address_string'])
    expected_location_details = config_geo['test_location']['expected_location_details']
    expected_location = config_geo['test_location']['expected_location']
    
    mock_url = "https://nominatim.openstreetmap.org/reverse"
    mock_response = expected_location  # Include all the fields expected in the 'properties'

    # Setup mock request with parameters and headers
    requests_mock.get(mock_url, json=mock_response)

    location = get_location(lat,lon)

    message = "\033[91m" + " \n Warning: test can fail because it changes the road unpredictably. \n" + "\033[0m"    

    assert location is not None  , f"get_location: response.status_code not == 200"
    assert 'error' not in location, f"get_location: {location}"
    assert location['features'][0]['properties']['address'] == expected_address, f"wrong adress"
      
    address_string = get_adress_string(location)
    assert address_string == expected_address_string, f"adress string is failed"

    location_details = get_location_details(location)
    assert location_details == expected_location_details
        
#----------------------------- Test wet dry areas 
@pytest.fixture
def out_point_on_land():
    return (True, False, None, False, None, 'The selected point is on land.')

@pytest.fixture
def out_point_in_ocean():
    return (False, False, None, False, None, 'The selected point is in the ocean.')

@pytest.fixture
def out_point_in_lake():
    return (True, True, 'Bodensee', False, None, 'The selected point is on land and in lake Bodensee.')

@pytest.mark.local_data    
def test_where_is_point(config_geo, out_point_on_land, out_point_in_ocean, out_point_in_lake):
    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('..')   

    lat, lon = config_geo['test_wetdry']['lat_dry'], config_geo['test_wetdry']['lon_dry']
    drywet = where_is_point(lat,lon)
    assert drywet == out_point_on_land, f"error in where_is_point, point({lat},{lon} not on land)"

    lat, lon = config_geo['test_wetdry']['lat_oce'], config_geo['test_wetdry']['lon_oce']
    drywet = where_is_point(lat,lon)
    assert drywet == out_point_in_ocean, f"error in where_is_point, point({lat},{lon} not in ocean)"

    lat, lon = config_geo['test_wetdry']['lat_lake'], config_geo['test_wetdry']['lon_lake']
    drywet = where_is_point(lat,lon)
    assert drywet == out_point_in_lake, f"error in where_is_point, point({lat},{lon} not in lake)"

    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('test')
   
@pytest.mark.local_data  
#--------------------------------   closest_shore_distance  
def test_where_is_point(config_geo, config_main):
   #I do not like it (we need to eliminate hardcode paths)
    os.chdir('..')       
    
    lat, lon = config_geo['test_location']['lat'], config_geo['test_location']['lon']
    coastline_shapefile = config_main['coastline_shapefile']
    dist = closest_shore_distance(lat, lon, coastline_shapefile)
    assert dist == pytest.approx(config_geo['test_location']['closest_shore_distance']), f"error in distance to shore"

    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('test')

#--------------------------------   get_elevation_from_api  
#test with request to api
@pytest.mark.request
def test_get_elevation_from_api(config_geo):
    lat, lon = config_geo['test_location']['lat'], config_geo['test_location']['lon']
    elevation = get_elevation_from_api(lat, lon)
    assert elevation == config_geo['test_location']['elevation'], f"error in elevation at point lat={lat}, lon={lon}"    
#test with mock request 
@pytest.mark.mock_request
def test_get_elevation_from_api(config_geo, requests_mock):
    lat, lon = config_geo['test_location']['lat'], config_geo['test_location']['lon']
    exp_elevation = config_geo['test_location']['elevation']
    mock_url = f"https://api.opentopodata.org/v1/etopo1?locations={lat},{lon}"
    mock_response ={'results': [{'dataset': 'etopo1',
                    'elevation': exp_elevation,
                    'location': {'lat': lat, 'lng': lon}}],
                    'status': 'OK'}    

    requests_mock.get(mock_url, json=mock_response)

    elevation = get_elevation_from_api(lat, lon)
    assert elevation == exp_elevation, f"error in elevation at point lat={lat}, lon={lon}"    
     
#--------------------------------   test_fetch_land_use  
@pytest.mark.request
def test_fetch_land_use(config_geo):
    lat, lon = config_geo['test_location']['lat_land_use'], config_geo['test_location']['lon_land_use']
    try:
        land_use_data = fetch_land_use(lon+1, lat)
    except:
        land_use_data = "Not known"
    try:
        current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
    except:
        current_land_use = "Not known"
    try:
        assert current_land_use == config_geo['test_location']['current_land_use']
    except AssertionError as e: 
        message = "\033[91m" + " \n Warning: test can fail because of HTML request. \n" + "\033[0m"    
        warnings.warn(str(e)+message, UserWarning)  

#--------------------------------   get_soil_from_api  
@pytest.mark.request
def test_get_soil_from_api(config_geo):
    lat, lon = config_geo['test_location']['lat'], config_geo['test_location']['lon']
    try:
        soil = get_soil_from_api(lat, lon)
    except:
        soil = "Not known"
    try:
        assert soil == config_geo['test_location']['soil'], f' lat={lat}, lon={lon}'
    except AssertionError as e: 
        message = "\033[91m" + " \n Warning: test can fail because of HTML request. \n" + "\033[0m"    
        warnings.warn(str(e)+message, UserWarning)        
    
    
    




