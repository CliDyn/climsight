"""
    Streamlit App Wrapper Module 
"""
#general 
import logging
import yaml
import os

#stgreamlit packeges
import streamlit as st
from streamlit_folium import st_folium
import folium

# climsight modules
from stream_handler import StreamHandler
from climsight_engine import llm_request, forming_request

logger = logging.getLogger(__name__)

def run_streamlit(config={}, api_key='', skip_llm_call=False):
    '''
        Inputs:
        - config (dict): Configuration, default is an empty dictionary.   
    - api_key (string): API Key, default is an empty string. if api_key='' (default) then skip_llm_call=True
    - skip_llm_call (bool): If True - skipp final call to LLM
        
    '''       
  
    # Config
    if not config:
        config_path = os.getenv('CONFIG_PATH', 'config.yml')
        logger.info(f"reading config from: {config_path}")
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {config_path}")
            raise RuntimeError(f"An error occurred while reading the file: {config_path}") from e
    try:
        model_name = config['model_name']
        climatemodel_name = config['climatemodel_name']
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
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")   

    if not isinstance(skip_llm_call, bool):
        logging.error(f"skip_llm_call must be bool in clim_request(...) ")
        raise TypeError("skip_llm_call must be  bool")    
    
    if not isinstance(api_key, str):
        logging.error(f"api_key must be a string in clim_request(...) ")
        raise TypeError("api_key must be a string")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment
            
    
    #read data while loading here 0000000000000000000000000000000000000000
    ##### like hist, future = load_data(config)

    clicked_coords = None
  

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
        
    # RUN submit button 
    if submit_button and user_message:
        if not api_key:
            api_key = api_key_input
        if not api_key:
            st.error("Please provide an OpenAI API key.")
            st.stop()

        with st.spinner("Getting info on a point..."):
            # Create a generator object by calling func2
            generator = forming_request(lat, lon, user_message)
            while True:
                try:
                    # Get the next intermediate result from the generator
                    result = next(generator)
                    st.markdown(f"{result}")
                except StopIteration as e:
                    # The generator is exhausted, and e.value contains the final result
                    content_message, input_params, config, df_data, figs, data = e.value
                    break            
###################################
#################################
#### Continue HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##################################
#################################
        with st.spinner("Generating..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box, display_method="write")

            if not isinstance(skip_llm_call, bool):
                logging.error(f"skip_llm_call must be bool in clim_request(...) ")
                raise TypeError("skip_llm_call must be  bool")    

            if not isinstance(api_key, str):
                logging.error(f"api_key must be a string in clim_request(...) ")
                raise TypeError("api_key must be a string")
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment
                if not api_key:        
                    skip_llm_call=True
                    api_key='Dummy' #for longchain api_key should be non empty str


            if not skip_llm_call:
            output = llm_request(content_message, input_params, config, api_key, stream_handler)   
            with open('output.txt', 'w') as file:
                # Write the content to the file
                print(output)
                file.write(output) 

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
                y=["Present Day Temperature", "Future Temperature"],
                color=["#d62728", "#2ca02c"],
            )
            st.markdown(
                "Precipitation (in mm)",
            )
            st.line_chart(
                df,
                x="Month",
                y=["Present Day Precipitation", "Future Precipitation"],
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
            # Determine the model information string based on climatemodel_name
            if climatemodel_name == 'AWI_CM':
                model_info = 'AWI-CM-1-1-MR, scenarios: historical and SSP5-8.5'
            elif climatemodel_name == 'tco1279':
                model_info = 'AWI-CM-3 TCo1279_DART, scenarios: historical (2000-2009) and SSP5-8.5 (2090-2099)'
            elif climatemodel_name == 'tco319':
                model_info = 'AWI-CM-3 TCo319_DART, scenarios: historical (2000-2009), and SSP5-8.5 (2090-2099)'
            else:
                model_info = 'unknown climate model'

            with st.expander("Source"):
                st.markdown(model_info)

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

        
        
    

    #call 
    #clim_request(lat, lon, user_message, stream_handler, data={}, config={}, api_key='', skip_llm_call=False)
    generator =  clim_request(lat, lon, user_message, stream_handler)

    while True:
        try:
            # Get the next intermediate result from the generator
            result = next(generator)
            print(f"Intermediate result: {result}")
        except StopIteration as e:
            # The generator is exhausted, and e.value contains the final result
            input_params, df_data, figs, data, config = e.value
            print(f"Input parameters:")      
            print(input_params)
            print(f"")      
            print(f"df_data:")      
            print(df_data)
            print(f"")      
            print(f"config:")      
            print(config)
            print(f"")     
            break   