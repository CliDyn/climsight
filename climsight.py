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
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from langchain.callbacks.base import BaseCallbackHandler


data_path = "./data/"
coastline_shapefile = "./data/natural_earth/coastlines/ne_50m_coastline.shp"
clicked_coords = None
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
# load population data from UN World Population Prospects 2022
pop_path = './population_data/WPP2022_Demographic_Indicators_Medium.csv'
pop_dat = pd.read_csv(pop_path)
# load natural hazard data from EM-DAT 
haz_path = './natural_hazard_data/public_emdat.xlsx'
nat_haz_dat = pd.read_excel(haz_path)

system_role = """
You are the system that should help people to evaluate the impact of climate change
on decisions they are taking today (e.g. install wind turbines, solar panels, build a building,
parking lot, open a shop, buy crop land). You are working with data on a local level,
and decisions also should be given for particular locations. You will be given information 
about changes in environmental variables for particular location, and how they will 
change in a changing climate. Your task is to provide assessment of potential risks 
and/or benefits for the planned activity related to change in climate. Use information 
about the country to retrieve information about policies and regulations in the 
area related to climate change, environmental use and activity requested by the user.
You don't have to use all variables provided to you, if the effect is insignificant,
don't use variable in analysis. DON'T just list information about variables, don't 
just repeat what is given to you as input. I don't want to get the code, 
I want to receive a narrative, with your assessments and advice. Format 
your response as MARKDOWN, don't use Heading levels 1 and 2.
"""

content_message = "{user_message} \n \
      Location: latitude = {lat}, longitude = {lon} \
      Adress: {location_str} \
      Policy: {policy} \
      Distance to the closest coastline: {distance_to_coastline} \
      Elevation above sea level: {elevation} \
      Current landuse: {current_land_use} \
      Current soil type: {soil} \
      Biodiversity: {biodiv} \
      Population: {population} \
      Current mean monthly temperature for each month: {hist_temp_str} \
      Future monthly temperatures for each month at the location: {future_temp_str}\
      Curent precipitation flux (mm/month): {hist_pr_str} \
      Future precipitation flux (mm/month): {future_pr_str} \
      Curent u wind component (in m/s): {hist_uas_str} \
      Future u wind component (in m/s): {future_uas_str} \
      Current v wind component (in m/s): {hist_vas_str} \
      Future v wind component (in m/s): {future_vas_str} \
      Natural hazards: {nat_hazards} \
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
def get_location(lat, lon):
    """
    Returns the address of a given latitude and longitude using the Nominatim geocoding service.

    Parameters:
    lat (float): The latitude of the location.
    lon (float): The longitude of the location.

    Returns:
    dict: A dictionary containing the address information of the location.
    """
    geolocator = Nominatim(user_agent="climsight")
    location = geolocator.reverse((lat, lon), language="en")
    return location.raw["address"]


@st.cache_data
def get_adress_string(location):
    """
    Returns a tuple containing two strings:
    1. A string representation of the location address with all the key-value pairs in the location dictionary.
    2. A string representation of the location address with only the country, state, city and road keys in the location dictionary.

    Parameters:
    location (dict): A dictionary containing the location address information.

    Returns:
    tuple: A tuple containing two strings.
    """
    location_str = "Adress: "
    for key in location:
        location_str = location_str + f"{key}:{location[key]}, "
    location_str_for_print = "**Address:** "
    if "country" in location:
        location_str_for_print += f"{location['country']}, "
    if "state" in location:
        location_str_for_print += f"{location['state']}, "
    if "city" in location:
        location_str_for_print += f"{location['city']}, "
    if "road" in location:
        location_str_for_print += f"{location['road']}"
    country = location.get("country", "")
    return location_str, location_str_for_print, country


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
    response = requests.get(url)
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
    response = requests.get(overpass_url, params={"data": overpass_query})
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
    url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=5"
    response = requests.get(url)
    data = response.json()
    return data["wrb_class_name"]


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
    response = requests.get(gbif_api_url, params=params)
    biodiv = response.json()
    return biodiv


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
            "Present day Temperature": hist_temp[0, 0, :],
            "Future Temeprature": future_temp[0, 0, :],
            "Present day Precipitation": hist_pr[0, 0, :],
            "Future Precipitation": future_pr[0, 0, :],
            "Present day Wind speed": np.hypot(hist_uas[0, 0, :], hist_vas[0, 0, :]),
            "Future Wind speed": np.hypot(future_uas[0, 0, :], future_vas[0, 0, :]),
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
def get_population_data(country):
    """
    Extracts population data (by UN) for a given country.

    Args:
    - country: Takes the country which is returned by the geolocator

    Returns:
    - red_pop_data (pandas.DataFrame): reduced DataFrame containing present day and future values for only the following variables:
        - TPopulation1July (as of 1 July, thousands)
        - PopDensity (as of 1 July, persons per square km)
        - PopGrowthRate (percentage)
        - LEx (Life Expactancy at Birth, both sexes, in years)
        - NetMigrations (Number of Migrants, thousands)    
    """
    today = datetime.date.today()
    current_year = today.year

    unique_locations = pop_dat['Location'].unique()
    my_location = country

    # check if data is available for the country that we are currently investigating
    if my_location in unique_locations:
        country_data = pop_dat[pop_dat['Location'] == country]
        red_pop_data = country_data[['Time', 'TPopulation1July', 'PopDensity', 'PopGrowthRate', 'LEx', 'NetMigrations']]

        fig, ax1 = plt.subplots(figsize=(10,6))
        plt.grid()

        # Total population data
        ax1.plot(red_pop_data['Time'], red_pop_data['TPopulation1July'], label='Total Population', color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('People in thousands', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # life expectancy
        ax2 = ax1.twinx()
        ax2.spines.right.set_position(('axes', 1.1))
        ax2.bar(red_pop_data['Time'], red_pop_data['LEx'], label='Life Expectancy', color='purple', alpha=0.1)
        ax2.set_ylabel('Life Expectancy in years', color='purple', )
        ax2.tick_params(axis='y', labelcolor='purple')

        # population growth data
        ax3 = ax1.twinx()
        ax3.plot(red_pop_data['Time'], red_pop_data['PopGrowthRate'], label='Population Growth Rate', color='green')
        ax3.set_ylabel('Population growth rate in %', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Net Migrations
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(('axes', 1.2))
        ax4.plot(red_pop_data['Time'], red_pop_data['NetMigrations'], label='Net Migrations', color='black', linestyle='dotted')
        ax4.set_ylabel('Net Migrations in thousands', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.axvline(x=current_year, color='orange', linestyle='--', label=current_year)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines+lines2+lines3+lines4, labels+labels2+labels3+labels4, loc='center right')

        plt.title(('Population changes in ' + country))
        return red_pop_data, fig 
    else:
        print(f"There is no UN population data available for {my_location}.")
        return None, None

@st.cache_data
def natural_hazard_data(country):
    """
    Extracts natural hazard data (by EM-DAT) for a given country.

    Args:
    - country: Takes the country which is returned by the geolocator

    Returns:
    - local_haz (pandas.DataFrame): DataFrame containing values for the following variables from 2001 until today:
        - DisNo. (unique 8-digit identifier including the year (4 digits) and a sequential number (4 digits) for each disaster event)
        - Classification Key (nique 15-character string identifying disasters in terms of the Group, Subgroup, Type and Subtype classification hierarchy)
        - Disaster Subgroup (geophysical, hydrological, meteorological, climatological, biological, extra-terrestrial)
        - Disaster Type
        - Disaster Subtype
        - Event Name
        - ISO
        - Country
        - Subregion
        - Region
        - Location
        - Origin
        - Associated Types (list of secondary disaster types)
        - Declaration (whether a state of emergency was declared by country)
        - Magnitude
        - Magnitude Scale
        - Start Year
        - Start Month
        - Start Day
        - End Year
        - End Month
        - End Day
        - Total Deaths
        - No. Injured
        - No. Affected
        - No. Homeless
        - Total Affacted (sum of previous 3)
        - Reconstruction Costs, Adjusted ('000 US$) (adjusted for inflation using Consumer Price Index)
        - Insured Damage, Adjusted ('000 US$)
        - Total Damage, Adjusted ('000  US$)
    """
    # create sub data set to only keep the relevant columnss
    selected_columns = [
        'DisNo.', 'Classification Key', 'Disaster Subgroup', 'Disaster Type',
        'Disaster Subtype', 'Event Name', 'ISO', 'Country', 'Subregion', 'Region',
        'Location', 'Origin', 'Associated Types', 'Declaration', 'Magnitude',
        'Magnitude Scale', 'Start Year', 'Start Month', 'Start Day', 'End Year',
        'End Month', 'End Day', 'Total Deaths', 'No. Injured', 'No. Affected',
        'No. Homeless', 'Total Affected', """Reconstruction Costs, Adjusted ('000 US$)""",
        """Insured Damage, Adjusted ('000 US$)""", """Total Damage, Adjusted ('000 US$)"""
    ]
    haz_dat = nat_haz_dat[selected_columns]

    unique_locs = haz_dat['Country'].unique()
    my_loc = country

    # check if data is available for the country that we are currently investigating
    if my_loc in unique_locs:
        local_haz = haz_dat[haz_dat['Country'] == country]

        # Filter out rows with NaN values
        local_haz_woNan = local_haz.dropna(subset=['Disaster Type'])

        # Color-blind friendly color palette
        colors = sns.color_palette('colorblind')
        sns.set(style='whitegrid')

        # Count occurrences of each disaster type
        disaster_counts = local_haz_woNan['Disaster Type'].value_counts()

        # get time span
        min = str(local_haz_woNan['Start Year'].min())
        max = str(local_haz['End Year'].max())

        # Plot for Disaster Type
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie(disaster_counts, labels=disaster_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, pctdistance=0.85, wedgeprops=dict(width=0.3), textprops=dict(rotation=25, va='center', fontsize=6))
        ax.set_title('Distribution of disaster types in ' + country + ' between ' + min + ' and ' + max, fontsize=8)
        return local_haz, fig
    
    else:
        print(f"There is no data available about natural hazards in {my_loc}.")
        return None, None

st.title(
    " :cyclone: \
         :ocean: :globe_with_meridians:  Climate Foresight"
)
# :umbrella_with_rain_drops: :earth_africa:  :tornado:

with st.form("query_form"):
    user_message = st.text_input(
        "Describe the activity that you would like to evaluate for this location:"
    )
    col1, col2 = st.columns(2)
    lat_default = 52.5240
    lon_default = 13.3700

    m = folium.Map(location=[lat_default, lon_default], zoom_start=13)
    with st.sidebar:
        api_key_input = st.text_input(
            "OpenAI API key",
            placeholder="Provide here or as OPENAI_API_KEY in your environment",
        )
        map_data = st_folium(m)
    if map_data:
        clicked_coords = map_data["last_clicked"]
        if clicked_coords:
            lat_default = clicked_coords["lat"]
            lon_default = clicked_coords["lng"]

    lat = col1.number_input("Latitude", value=lat_default, format="%.4f")
    lon = col2.number_input("Longitude", value=lon_default, format="%.4f")

    api_key = api_key_input or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Please provide an OpenAI API key.")
        st.stop()    

    generate_button = st.form_submit_button("Generate")

if generate_button and user_message:
# if st.button("Generate") and user_message:
    with st.spinner("Getting info on a point..."):
        location = get_location(lat, lon)
        location_str, location_str_for_print, country = get_adress_string(location)
        st.markdown(f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}")
        st.markdown(location_str_for_print)
        elevation = get_elevation_from_api(lat, lon)
        st.markdown(f"**Elevation:** {elevation} m")

        population, pop_fig = get_population_data(country)
        #print(population)
        st.markdown("**Population:**")
        st.pyplot(pop_fig)
        st.text(
            "Source: UN World Population Prospect 2022"
        )

        land_use_data = fetch_land_use(lon, lat)
        try:
            current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
        except:
            current_land_use = "Not known"
        st.markdown(f"**Current land use:** {current_land_use}")

        soil = get_soil_from_api(lat, lon)
        st.markdown(f"**Soil type:** {soil}")

        biodiversity = fetch_biodiversity(round(lat), round(lon))
        biodiv_set = set()
        for record in biodiversity['results']:
            if record.get('taxonRank') != 'UNRANKED':
                biodiv_set.add(record['scientificName'])
        biodiv = list(biodiv_set)
        if biodiv == []:
            biodiv = "Not known"
        print(biodiv)
        st.markdown(f"**Biodiversity:** {biodiv}")

        distance_to_coastline = closest_shore_distance(lat, lon, coastline_shapefile)
        st.markdown(f"**Distance to the shore:** {round(distance_to_coastline, 2)} m")

        # create pandas dataframe
        df, data_dict = extract_climate_data(lat, lon, hist, future)
        # Plot the chart
        st.text(
            "Near surface temperature [source: AWI-CM-1-1-MR, historical and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present day Temperature", "Future Temeprature"],
            color=["#d62728", "#2ca02c"],
        )
        st.text(
            "Precipitation [source: AWI-CM-1-1-MR, historical and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present day Precipitation", "Future Precipitation"],
            color=["#d62728", "#2ca02c"],
        )
        st.text(
            "Wind speed [source: AWI-CM-1-1-MR, historical and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present day Wind speed", "Future Wind speed"],
            color=["#d62728", "#2ca02c"],
        )

        nat_hazards, haz_fig = natural_hazard_data(country)
        #print(nat_hazards)
        st.markdown("**Natural hazards:**")
        st.pyplot(haz_fig)
        st.text(
            "Source: EM-DAT"
        )
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
            policy=policy,
            distance_to_coastline=str(distance_to_coastline),
            elevation=str(elevation),
            current_land_use=current_land_use,
            soil=soil,
            biodiv=biodiv,
            population = population,
            hist_temp_str=data_dict["hist_temp"],
            future_temp_str=data_dict["future_temp"],
            hist_pr_str=data_dict["hist_pr"],
            future_pr_str=data_dict["future_pr"],
            hist_uas_str=data_dict["hist_uas"],
            future_uas_str=data_dict["future_uas"],
            hist_vas_str=data_dict["hist_vas"],
            future_vas_str=data_dict["future_vas"],
            nat_hazards = nat_hazards,
            verbose=True,
        )