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


data_path = "./data/"
coastline_shapefile = "./data/natural_earth/coastlines/ne_50m_coastline.shp"
clicked_coords = None
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"

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
      Current mean monthly temperature for each month: {hist_temp_str} \
      Future monthly temperatures for each month at the location: {future_temp_str}\
      Curent precipitation flux (mm/month): {hist_pr_str} \
      Future precipitation flux (mm/month): {future_pr_str} \
      Curent u wind component (in m/s): {hist_uas_str} \
      Future u wind component (in m/s): {future_uas_str} \
      Curent v wind component (in m/s): {hist_vas_str} \
      Future v wind component (in m/s): {future_vas_str} \
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
    return location_str, location_str_for_print


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

st.title(
    " :cyclone: \
         :ocean: :globe_with_meridians:  Climate Foresight"
)
# :umbrella_with_rain_drops: :earth_africa:  :tornado:
user_message = st.text_input(
    "Describe activity you would like to evaluate for this location:"
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


if st.button("Generate") and user_message:
    with st.spinner("Getting info on a point..."):
        location = get_location(lat, lon)
        location_str, location_str_for_print = get_adress_string(location)
        st.markdown(f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}")
        st.markdown(location_str_for_print)
        elevation = get_elevation_from_api(lat, lon)
        st.markdown(f"**Elevation:** {elevation} m")

        land_use_data = fetch_land_use(lon, lat)
        try:
            current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
        except:
            current_land_use = "Not known"
        st.markdown(f"**Current land use:** {current_land_use}")

        soil = get_soil_from_api(lat, lon)
        st.markdown(f"**Soil type:** {soil}")

        distance_to_coastline = closest_shore_distance(lat, lon, coastline_shapefile)
        st.markdown(f"**Distance to the shore:** {round(distance_to_coastline, 2)} m")

        # create pandas dataframe
        df, data_dict = extract_climate_data(lat, lon, hist, future)
        # Plot the chart
        st.text(
            "Near surface temperature [souce: AWI-CM-1-1-MR, historical and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present day Temperature", "Future Temeprature"],
            color=["#d62728", "#2ca02c"],
        )
        st.text(
            "Precipitation [souce: AWI-CM-1-1-MR, historical and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present day Precipitation", "Future Precipitation"],
            color=["#d62728", "#2ca02c"],
        )
        st.text(
            "Wind speed [souce: AWI-CM-1-1-MR, historical and SSP5-8.5]",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present day Wind speed", "Future Wind speed"],
            color=["#d62728", "#2ca02c"],
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
            hist_temp_str=data_dict["hist_temp"],
            future_temp_str=data_dict["future_temp"],
            hist_pr_str=data_dict["hist_pr"],
            future_pr_str=data_dict["future_pr"],
            hist_uas_str=data_dict["hist_uas"],
            future_uas_str=data_dict["future_uas"],
            hist_vas_str=data_dict["hist_vas"],
            future_vas_str=data_dict["future_vas"],
            verbose=True,
        )
