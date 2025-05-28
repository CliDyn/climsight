"""
    Streamlit App Wrapper Module 
"""
#general 
import logging
import yaml
import os
import pandas as pd

#streamlit packeges
import streamlit as st
from streamlit_folium import st_folium
import folium

# rag
from rag import load_rag 

# climsight modules
from stream_handler import StreamHandler
from data_container import DataContainer
from climsight_engine import normalize_longitude, llm_request, forming_request, location_request
from extract_climatedata_functions import plot_climate_data

#ui for saving docs
from datetime import datetime
from ui import prepare_download_content, prepare_pdf_content

logger = logging.getLogger(__name__)

data_pocket = DataContainer()

def run_streamlit(config, api_key='', skip_llm_call=False, rag_activated=True, embedding_model='', chroma_path='', references=None):
    """
    Runs the Streamlit interface for ClimSight, allowing users to interact with the system.
    Args:
        - config (dict): Configuration, default is an empty dictionary.   
        - api_key (string): API Key, default is an empty string.
        - skip_llm_call (bool): If True - skip final call to LLM
        - rag_activated (bool): whether or not to include the text based rag
        - embedding_model (str): embedding model to be used for loading the Chroma database.
        - chroma_path (str): Path where the Chroma database is stored.
        - references (dict): References for the data used in the analysis.
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

    api_key_local = os.environ.get("OPENAI_API_KEY_LOCAL")
    if not api_key_local:
        api_key_local = ""
        
    #read data while loading here 
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
        #with st.form(key='side_form'):
        #    location_button = st.form_submit_button(label='Get location info')
        #    if location_button:
        #        st.markdown()
        map_data = st_folium(m)
            
    if map_data:
        clicked_coords = map_data["last_clicked"]
        if clicked_coords:
            lat_default = clicked_coords["lat"]
            lon_default = clicked_coords["lng"]
        # normalize longitude in case map has been move around global (more than) once
        lon_default = normalize_longitude(lon_default)

    # Wrap the input fields and the submit button in a form
    with st.form(key='my_form'):
        user_message = st.text_input(
            "Describe the activity that you would like to evaluate for this location:",
            placeholder="Are the conditions suitable for setting up a solar panel field in this area over the next decade?",
        )
        col1, col2 = st.columns(2)
        lat = col1.number_input("Latitude", value=lat_default, format="%.4f")
        lon = col2.number_input("Longitude", value=lon_default, format="%.4f")

        # Predefined options
        options = ["gpt-4.1-nano", "gpt-4o","o3-mini","gpt-4.1", "o1", "o3"]
        # Determine the default value and modify the options list if needed
        default_model = config.get('model_name_combine_agent')
        if default_model:
            if default_model not in options:
                options.insert(0, default_model)
                default_index = 0
            else:
                default_index = options.index(default_model)
        else:
            default_index = 1
        col1, col2 = st.columns([1, 1])
        with col2:
            config['model_name_combine_agent'] = st.selectbox(
                "Model for synthesis:",
                options,
                index=default_index
            )            
        with col1:         
            show_add_info = st.toggle("Provide additional information", value=False, help="""If this is activated you will see all the variables
                                    that were taken into account for the analysis as well as some plots.""")
            smart_agent   = st.toggle("Use smart agent feature", value=False, help="""If this is activated together with Agent mode, ClimSight will make additional requests to Wikipedia and RAG, which can significantly increase response time.""")
            # remove the llmModeKey_box from the form, as we tend to run the agent mode, direct mode is for development only
            #llmModeKey_box = st.radio("Select LLM mode 👉", key="visibility", options=["Direct", "Agent (experimental)"])
            # Include the API key input within the form only if it's not found in the environment
        if (not api_key) and config['model_type'] == "openai":
            api_key_input = st.text_input(
                "OpenAI API key",
                placeholder="Enter your OpenAI API key here",
                type="password",
            )

       
        # Replace the st.button with st.form_submit_button
        submit_button = st.form_submit_button(label='Generate')
        
    # RUN submit button - ANALYSIS LOGIC ONLY (NO DISPLAY HERE!)
    if submit_button and user_message:
        if not api_key:
            api_key = api_key_input
        if (not api_key) and (not skip_llm_call) and (config['model_type'] == "openai"):
            st.error("Please provide an OpenAI API key.")
            st.stop()
        # Update config with the selected LLM mode
        #config['llmModeKey'] = "direct_llm" if llmModeKey_box == "Direct" else "agent_llm"    
        config['show_add_info'] = show_add_info
        config['use_smart_agent'] = smart_agent
        
        # Creating a potential bottle neck here with loading the db inside the streamlit form, but it works fine 
        # for the moment. Just making a note here for any potential problems that might arise later one. 
        # Load RAG
        if not skip_llm_call and rag_activated:
            chroma_path_ipcc = chroma_path[0]
            chroma_path_general = chroma_path[1]
            try:
                logger.info("RAG is activated and skipllmcall is False. Loading IPCC RAG database...")
                ipcc_rag_ready, ipcc_rag_db = load_rag(embedding_model, chroma_path_ipcc, api_key) # load the RAG database 
            except Exception as e:
                st.error(f"Loading of the IPCC RAG database failed unexpectedly, please check the logs. {e}")
                logger.warning(f"IPCC RAG database initialization skipped or failed: {e}")
                ipcc_rag_ready = False
                ipcc_rag_db = None
            try:
                logger.info("RAG is activated and skipllmcall is False. Loading general RAG database...")
                general_rag_ready, general_rag_db = load_rag(embedding_model, chroma_path_general, api_key) # load the RAG database 
            except Exception as e:
                st.error(f"Loading of the (general) RAG database failed unexpectedly, please check the logs. {e}")
                logger.warning(f"(General) RAG database initialization skipped or failed: {e}")
                general_rag_ready = False
                general_rag_db = None
                 
        is_on_land = True

        if config['llmModeKey'] == "direct_llm":
            # Call the forming_request function
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
                            data_pocket.df['df_data'] = df_data
                            data_pocket.figs = figs
                            data_pocket.data = data
                        break            
        else:
            # Agent LLM mode (load only location info)
            with st.spinner("Getting info on a point..."):
                st.markdown(f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}")
                # get first location information only, input_params and content_message are only partly filled
                content_message, input_params = location_request(config, lat, lon)
                if not input_params:
                    is_on_land = False
                    st.markdown(f"The selected point is in the ocean.\n Please choose a location on land.")
                else:
                    # extend input_params with user_message
                    input_params['user_message'] = user_message
                    content_message = "Human request: {user_message} \n " + content_message
                    st.markdown(f"{input_params['location_str_for_print']}")
                    if input_params['is_inland_water']:
                        st.markdown(f"""{input_params['water_body_status']}: Our analyses are currently only meant for land areas. Please select another location for a better result.""")

        # SIMPLIFIED PROCESSING - NO DISPLAY HERE!
        if is_on_land:        
            progress_area = st.empty()  # This will display progress updates
            
            # Create temporary containers for streaming (not for final display)
            temp_result_container = st.empty()
            temp_reference_container = st.empty()
            
            # Initialize the spinner outside, but we'll use our own progress messages
            with st.spinner("Processing your request..."):
                # Create StreamHandler with temporary components
                stream_handler = StreamHandler(temp_result_container, temp_reference_container)

                # Add a method to update the progress area
                def update_progress_ui(message):
                    progress_area.info(message)

                # Attach this method to your StreamHandler
                stream_handler.update_progress = update_progress_ui

                # Now call llm_request with this enhanced stream_handler
                if not skip_llm_call:
                    output, input_params, content_message = llm_request(
                        content_message, input_params, config, api_key, api_key_local, 
                        stream_handler, ipcc_rag_ready, ipcc_rag_db, 
                        general_rag_ready, general_rag_db, data_pocket,
                        references=references
                    )
            
            # Clear the progress area and temporary containers after completion
            progress_area.empty()
            temp_result_container.empty()
            temp_reference_container.empty()
            
            # Store results in session state
            if not skip_llm_call and 'output' in locals() and output:
                st.session_state['last_output'] = output
                st.session_state['last_input_params'] = input_params
                st.session_state['last_figs'] = data_pocket.figs
                st.session_state['last_references'] = references
                st.session_state['last_show_add_info'] = show_add_info
                st.session_state['last_climatemodel_name'] = climatemodel_name
                
    # DISPLAY LOGIC - OUTSIDE OF SUBMIT BUTTON BLOCK
    # This will show the report whenever it exists in session state
    if 'last_output' in st.session_state and st.session_state['last_output']:
        # Get show_add_info from session state
        show_add_info_display = st.session_state.get('last_show_add_info', False)
        
        if show_add_info_display:
            tab_text, tab_add, tab_refs = st.tabs(["Report", "Additional information", "References"])
        else:
            tab_text, tab_refs = st.tabs(["Report", "References"])
        
        with tab_text:
            st.markdown(st.session_state['last_output'])
        
        with tab_refs:
            if 'last_references' in st.session_state and st.session_state['last_references']:
                for ref in st.session_state['last_references'].get('used', []):
                    st.markdown(f"- {ref}")
        
        if show_add_info_display:
            with tab_add:
                stored_input_params = st.session_state.get('last_input_params', {})
                stored_figs = st.session_state.get('last_figs', {})
                stored_climatemodel_name = st.session_state.get('last_climatemodel_name', 'unknown')
                
                st.subheader("Additional information", divider='rainbow')
                if 'lat' in stored_input_params and 'lon' in stored_input_params:
                    st.markdown(f"**Coordinates:** {stored_input_params['lat']}, {stored_input_params['lon']}")
                if 'elevation' in stored_input_params:
                    st.markdown(f"**Elevation:** {stored_input_params['elevation']} m")
                if 'current_land_use' in stored_input_params:  
                    st.markdown(f"**Current land use:** {stored_input_params['current_land_use']}")
                if 'soil' in stored_input_params:
                    st.markdown(f"**Soil type:** {stored_input_params['soil']}")
                if 'biodiv' in stored_input_params:    
                    st.markdown(f"**Occuring species:** {stored_input_params['biodiv']}")
                if 'distance_to_coastline' in stored_input_params:  
                    st.markdown(f"**Distance to the shore:** {round(float(stored_input_params['distance_to_coastline']), 2)} m")
                
                # Climate Data
                if (not config.get('use_high_resolution_climate_model', False)) and ('df_data' in data_pocket.df and data_pocket.df['df_data'] is not None):
                    df_data = data_pocket.df['df_data']
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
                    if stored_climatemodel_name == 'AWI_CM':
                        model_info = 'AWI-CM-1-1-MR, scenarios: historical and SSP5-8.5'
                    elif stored_climatemodel_name == 'tco1279':
                        model_info = 'AWI-CM-3 TCo1279_DART, scenarios: historical (2000-2009) and SSP5-8.5 (2090-2099)'
                    elif stored_climatemodel_name == 'tco319':
                        model_info = 'AWI-CM-3 TCo319_DART, scenarios: historical (2000-2009), and SSP5-8.5 (2090-2099)'
                    else:
                        model_info = 'unknown climate model'

                    with st.expander("Source"):
                        st.markdown(model_info)

                if config.get('use_high_resolution_climate_model', False):
                    to_plot = False
                    try:                             
                        df_list = data_pocket.data['high_res_climate']['df_list']
                        to_plot = True
                    except Exception as e:
                        logger.warning(f"Error by getting high resolution climate data from data pocket: {e}")
                    if to_plot:    
                        figs_climate = plot_climate_data(df_list)
                        st.markdown("**Climate data:**")
                        for fig_dict in figs_climate:
                            st.pyplot(fig_dict['fig'])
                        
                        with st.expander("Source"):
                            st.markdown(figs_climate[0]['source'])

                # Natural Hazards
                if 'haz_fig' in stored_figs:
                    st.markdown("**Natural hazards:**")
                    st.pyplot(stored_figs['haz_fig']['fig'])
                    with st.expander("Source"):
                        st.markdown(stored_figs['haz_fig']['source'])

                # Population Data
                if 'population_plot' in stored_figs:
                    st.markdown("**Population Data:**")
                    st.pyplot(stored_figs['population_plot']['fig'])
                    with st.expander("Source"):
                        st.markdown(stored_figs['population_plot']['source'])
        
        # Download buttons
        st.markdown("---")  # Add a separator
        
        # Get data from session state
        stored_output = st.session_state['last_output']
        stored_input_params = st.session_state.get('last_input_params', {})
        stored_figs = st.session_state.get('last_figs', {})
        stored_references = st.session_state.get('last_references', {})
        
        # Prepare both text and PDF content
        download_content_text = prepare_download_content(
            stored_output, stored_input_params, stored_figs, data_pocket, stored_references
        )
        
        download_content_pdf = prepare_pdf_content(
            stored_output, stored_input_params, stored_figs, data_pocket, stored_references
        )
        
        # Create download buttons with centered layout
        col1, col2, col3, col4, col5 = st.columns([1, 2, 0.5, 2, 1])
        with col2:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_pdf = f"climsight_report_{timestamp}.pdf"
            
            st.download_button(
                label="📥 Download PDF Report",
                data=download_content_pdf,
                file_name=filename_pdf,
                mime="application/pdf",
                help="Download the complete analysis report as a PDF file",
                key=f"download_button_pdf_{timestamp}"
            )
        
        with col4:
            filename_txt = f"climsight_report_{timestamp}.txt"
            
            st.download_button(
                label="📄 Download Text Report",
                data=download_content_text,
                file_name=filename_txt,
                mime="text/plain",
                help="Download the complete analysis report as a text file",
                key=f"download_button_txt_{timestamp}"
            )
                            
    return