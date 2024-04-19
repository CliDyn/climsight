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
import sys

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
from economic_functions import (
   get_population,
   plot_population,
   x_year_mean_population   
)

from climate_functions import (
   #load_config,
   load_data,
   extract_climate_data
)
from environmental_functions import (
   fetch_biodiversity,
   load_nat_haz_data,
   filter_events_within_square,
   plot_disaster_counts
)

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

model_name = config['model_name']
data_path = config['data_settings']['data_path']
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

# Check if 'skipLLMCall' argument is provided
skip_llm_call = 'skipLLMCall' in sys.argv

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

hist, future = load_data(config)

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
    show_add_info = st.toggle("Provide additional information", value=False, help="""If this is activated you will see all the variables
                               that were taken into account for the analysis as well as some plots.""")
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

        # CALCULATIONS / RUNNING FUNCTIONS
        location = get_location(lat, lon)
        location_str, location_str_for_print, country = get_adress_string(location)

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

        try:
            land_use_data = fetch_land_use(lon, lat)
        except:
            land_use_data = "Not known"
        try:
            current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
        except:
            current_land_use = "Not known"

        try:
            soil = get_soil_from_api(lat, lon)
        except:
            soil = "Not known"

        biodiv = fetch_biodiversity(round(lat), round(lon))

        distance_to_coastline = closest_shore_distance(lat, lon, coastline_shapefile)

        # create pandas dataframe
        df, data_dict = extract_climate_data(lat, lon, hist, future, config)

        filtered_events_square, promt_hazard_data = filter_events_within_square(lat, lon, haz_path, distance_from_event)

        population = x_year_mean_population(pop_path, country, year_step=year_step, start_year=start_year, end_year=end_year)

        haz_fig = plot_disaster_counts(filtered_events_square)
        population_plot = plot_population(pop_path, country)

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
        # Only proceed with the LLM call if 'skipLLMCall' is NOT provided
        if not skip_llm_call:  
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
                hist_temp_str=data_dict["hist_t2m"],
                future_temp_str=data_dict["future_t2m"],
                hist_pr_str=data_dict["hist_precip"],
                future_pr_str=data_dict["future_precip"],
                hist_uas_str=data_dict["hist_u_wind"],
                future_uas_str=data_dict["future_u_wind"],
                hist_vas_str=data_dict["hist_v_wind"],
                future_vas_str=data_dict["future_v_wind"],
                nat_hazards = promt_hazard_data,
                population = population,
                verbose=True,
            )

    # PLOTTING ADDITIONAL INFORMATION
    if show_add_info: 
        st.subheader("Additional information", divider='rainbow')
        st.markdown(f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}")
        st.markdown(location_str_for_print) 
        st.markdown(f"**Elevation:** {elevation} m")
        st.markdown(f"**Current land use:** {current_land_use}")
        st.markdown(f"**Soil type:** {soil}")
        st.markdown(f"**Occuring species:** {biodiv}")
        st.markdown(f"**Distance to the shore:** {round(distance_to_coastline, 2)} m")
        
        # Climate Data
        st.markdown("**Climate data:**")
        st.markdown(
            "Near surface temperature (in °C)",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present Day T2m", "Future T2m"],
            color=["#d62728", "#2ca02c"],
        )
        st.markdown(
            "Precipitation (in mm)",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present Day Precip", "Future Precip"],
            color=["#d62728", "#2ca02c"],
        )
        st.markdown(
            "Wind speed (in m*s-1)",
        )
        st.line_chart(
            df,
            x="Month",
            y=["Present Day Wind Speed", "Future Wind Speed"],
            color=["#d62728", "#2ca02c"],
        )
        with st.expander("Source"):
            st.markdown('AWI-CM-1-1-MR, historical, and SSP5-8.5')

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
