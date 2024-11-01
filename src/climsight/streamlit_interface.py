"""
    Streamlit App Wrapper Module 
"""
#general 
import logging
import yaml
import os

#streamlit packeges
import streamlit as st
from streamlit_folium import st_folium
import folium

# rag
from rag import load_rag 

# climsight modules
from stream_handler import StreamHandler
from climsight_engine import llm_request, forming_request

logger = logging.getLogger(__name__)

def run_streamlit(config, api_key='', skip_llm_call=False, rag_activated=True, embedding_model='', chroma_path=''):
    """
    Runs the Streamlit interface for ClimSight, allowing users to interact with the system.
    Args:
        - config (dict): Configuration, default is an empty dictionary.   
        - api_key (string): API Key, default is an empty string.
        - skip_llm_call (bool): If True - skip final call to LLM
        - rag_activated (bool): whether or not to include the text based rag
        - embedding_model (str): embedding model to be used for loading the Chroma database.
        - chroma_path (str): Path where the Chroma database is stored.
    Returns:
        None
    """     
  
    # Config
    try:
        climatemodel_name = config['climatemodel_name']
        lat_default = config['lat_default']
        lon_default = config['lon_default']
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")   

    if not isinstance(skip_llm_call, bool):
        logging.error(f"skip_llm_call must be bool ")
        raise TypeError("skip_llm_call must be  bool")    
    
    if not isinstance(api_key, str):
        logging.error(f"api_key must be a string ")
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
        llmModeKey_box = st.radio("Select LLM mode 👉", key="visibility", options=["Direct", "Agent (experimental)"])
    
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
            if (not api_key) and (not skip_llm_call):
                st.error("Please provide an OpenAI API key.")
                st.stop()
            # Update config with the selected LLM mode
            config['llmModeKey'] = "direct_llm" if llmModeKey_box == "Direct" else "agent_llm"    
            
            # Creating a potential bottle neck here with loading the db inside the streamlit form, but it works fine 
            # for the moment. Just making a note here for any potential problems that might arise later one. 
            # Load RAG
            if not skip_llm_call and rag_activated:
                try:
                    logger.info("RAG is activated and skipllmcall is False. Loading RAG database...")
                    rag_ready, rag_db = load_rag(embedding_model, chroma_path, api_key) # load the RAG database 
                except Exception as e:
                    st.error(f"Loading of the RAG database failed unexpectedly, please check the logs. {e}")
                    logger.warning(f"RAG database initialization skipped or failed: {e}")
                    rag_ready = False
                    rag_db = None
                 
            is_on_land = True
            with st.spinner("Getting info on a point..."):
                # Create a generator object by calling func2
                generator = forming_request(config, lat, lon, user_message)
                while True:
                    try:
                        # Get the next intermediate result from the generator
                        result = next(generator)
                        st.markdown(f"{result}")
                    except StopIteration as e:
                        # The generator is exhausted, and e.value contains the final result
                        
                        gen_output = e.value
                        # check if Error ocure:
                        if isinstance(gen_output,str):
                            if "Error" in gen_output:
                                if "point_is_in_ocean" in gen_output:
                                    is_on_land = False
                                    st.markdown(f"The selected point is in the ocean.\n Please choose a location on land.")
                        else:    
                            content_message, input_params, df_data, figs, data = e.value
                        break            
            if is_on_land:        
                with st.spinner("Generating..."):
                    chat_box = st.empty()
                    stream_handler = StreamHandler(chat_box, display_method="write")
                    if not skip_llm_call:
                        output = llm_request(content_message, input_params, config, api_key, stream_handler, rag_ready, rag_db)   

                    # PLOTTING ADDITIONAL INFORMATION
                    if show_add_info: 
                        st.subheader("Additional information", divider='rainbow')
                        st.markdown(f"**Coordinates:** {input_params['lat']}, {input_params['lon']}")
                        st.markdown(f"**Elevation:** {input_params['elevation']} m")
                        st.markdown(f"**Current land use:** {input_params['current_land_use']}")
                        st.markdown(f"**Soil type:** {input_params['soil']}")
                        st.markdown(f"**Occuring species:** {input_params['biodiv']}")
                        st.markdown(f"**Distance to the shore:** {round(float(input_params['distance_to_coastline']), 2)} m")
                        
                        # Climate Data
                        st.markdown("**Climate data:**")
                        st.markdown(
                            "Near surface temperature (in °C)",
                        )
                        st.line_chart(
                            df_data,
                            x="Month",
                            y=["Present Day Temperature", "Future Temperature"],
                            color=["#1f77b4", "#d62728"],
                        )
                        st.markdown(
                            "Precipitation (in mm)",
                        )
                        st.line_chart(
                            df_data,
                            x="Month",
                            y=["Present Day Precipitation", "Future Precipitation"],
                            color=["#1f77b4", "#d62728"],
                        )
                        st.markdown(
                            "Wind speed (in m*s-1)",
                        )
                        st.line_chart(
                            df_data,
                            x="Month",
                            y=["Present Day Wind Speed", "Future Wind Speed"],
                            color=["#1f77b4", "#d62728"],
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
                        if 'haz_fig' in figs:
                            st.markdown("**Natural hazards:**")
                            st.pyplot(figs['haz_fig']['fig'])
                            with st.expander("Source"):
                                st.markdown(figs['haz_fig']['source'])

                        # Population Data
                        if 'population_plot' in figs:
                            st.markdown("**Population Data:**")
                            st.pyplot(figs['population_plot']['fig'])
                            with st.expander("Source"):
                                st.markdown(figs['population_plot']['source'])
                            
    return
