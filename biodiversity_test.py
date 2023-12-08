import requests

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
        #"radius": 10,  # Adjust the radius as needed
        #"limit": 10,  # Adjust the limit as needed
    }
    response = requests.get(gbif_api_url, params=params)
    data = response.json()

    return data

if __name__ == "__main__":
    # Replace these coordinates with the ones you want to test
    test_longitude = 10
    test_latitude = 51

    biodiversity_data = fetch_biodiversity(test_longitude, test_latitude)

    # Print the entire response for exploration
    # print(biodiversity_data)

# Extract scientific names
#animal_names = [record['scientificName'] for record in biodiversity_data['results'] if record.get('taxonRank') != 'UNRANKED']

animal_names_set = set()

for record in biodiversity_data['results']:
    if record.get('taxonRank') != 'UNRANKED':
        animal_names_set.add(record['scientificName'])

animal_names = list(animal_names_set)
#print(animal_names)
# Print the names
#for name in animal_names:
    #print(name)



############## Population UN Data
import pycountry

def get_country_codes(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_2, country.alpha_3
        else:
            raise ValueError("Country not found")
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Example usage:
country_name = "Germany"
iso2 = get_country_codes(country_name)

print(f"Country: {country_name}")
print(f"ISO2 Code: {iso2}")

import pandas as pd
import requests
import json

baseurl = "https://population.un.org/dataportalapi/api/v1"

target = baseurl + "/data/indicators/50,76,49,54,60,57/locations/{iso2}/start/1950/end/2100" 


# Get the response, which includes the first page of data as well as information on pagination and number of records
response = requests.get(target)

# Converts call into JSON
j = response.json()

# Converts JSON into a pandas DataFrame.
df = pd.json_normalize(j['data']) # pd.json_normalize flattens the JSON to accomodate nested lists within the JSON structure

# Loop until there are new pages with data
while j['nextPage'] != None:
    # Reset the target to the next page
    target = j['nextPage']

    #call the API for the next page
    response = requests.get(target)

    # Convert response to JSON format
    j = response.json()

    # Store the next page in a data frame
    df_temp = pd.json_normalize(j['data'])

    # Append next page to the data frame
    df = df.append(df_temp)
print(df)