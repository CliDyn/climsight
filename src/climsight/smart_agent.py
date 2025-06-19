# smart_agent.py

from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import netCDF4 as nc
import numpy as np
import os
import ast
from typing import Union

from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import AIMessage
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
#import requests
#from bs4 import BeautifulSoup
#from urllib.parse import quote_plus
#from langchain.schema import Document
from langchain.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from aitta_client import Model, Client
from aitta_client.authentication import APIKeyAccessTokenSource

# Import AgentState from climsight_classes
from climsight_classes import AgentState
import calendar
import pandas as pd

def get_aitta_chat_model(model_name, **kwargs):
    aitta_url = 'https://api-climatedt-aitta.2.rahtiapp.fi'
    aitta_api_key = os.environ['AITTA_API_KEY']
    client = Client(aitta_url, APIKeyAccessTokenSource(aitta_api_key, aitta_url))
    model = Model.load(model_name, client)
    access_token = client.access_token_source.get_access_token()
    return ChatOpenAI(
        openai_api_key=access_token,
        openai_api_base=model.openai_api_url,
        model_name=model.id,
        **kwargs
    )

def smart_agent(state: AgentState, config, api_key, api_key_local, stream_handler):
#def smart_agent(state: AgentState, config, api_key):
    stream_handler.update_progress("Running advanced analysis with smart agent...")
    lat = float(state.input_params['lat'])
    lon = float(state.input_params['lon'])
    temperature = 0
    if "o1" in config['model_name_tools']:
        temperature = 1

    # System prompt
    prompt = f"""
    You are the smart agent of ClimSight. Your task is to retrieve necessary components of the climatic datasets based on the user's request.
    You have access to tools called "get_data_components", "wikipedia_search", "RAG_search" and "ECOCROP_search" which you can use to retrieve the necessary environmental data components.
    - "get_data_components" will retrieve the necessary data from the climatic datasets at the location of interest (latitude: {lat}, longitude: {lon}). It accepts an 'environmental_data' parameter to specify the type of data, and a 'months' parameter to specify which months to retrieve data for. The 'months' parameter is a list of month names (e.g., ['Jan', 'Feb', 'Mar']). If 'months' is not specified, data for all months will be retrieved.
    <Important> Call "get_data_components" tool multiple times if necessary, but only within one iteration, [chat_completion -> n * "get_data_components" -> chat_completion] after you recieve the necessary data from wikipedia_search and RAG_search. </Important>
    - "wikipedia_search" will help you determine the necessary data to retrieve with the get_data_components tool.
    - "RAG_search" can provide detailed information about environmental conditions for growing corn from your internal knowledge base.
    - "ECOCROP_search" will help you determine the specific environmental requirements for the crop of interest from ecocrop database.
    call "ECOCROP_search" ONLY and ONLY if you sure that the user question is related to the crop of interest.
    """
    if config['model_type'] in ("local", "aitta"):
        prompt += f"""

        <Important> Always call the wikipedia_search, RAG_search, and ECOCROP_search tools as needed, but only one at a time per turn.; it will help you determine the necessary data to retrieve with the get_data_components tool. At second step, call the get_data_components tool with the necessary data.</Important>
        """
    else:
        prompt += f"""

        <Important> ALWAYS call FIRST SIMULTANIOUSLY the wikipedia_search, RAG_search and "ECOCROP_search"; it will help you determine the necessary data to retrieve with the get_data_components tool. At second step, call the get_data_components tool with the necessary data.</Important>
        """        
    prompt += f"""

    Use these tools to get the data you need to answer the user's question.
    After retrieving the data, provide a concise summary of the parameters you retrieved, explaining briefly why they are important. Keep your response short and to the point.
    Do not include any additional explanations or reasoning beyond the concise summary.
    Do not include any chain-of-thought reasoning or action steps in your final answer.
    Do not ask the user for any additional information, but you can include into the final answer what kind of information user should provide in the future.
    
    <Important> 
    For the final response try to follow the following format:
    'The [retrieved values of the parameter] for the [object of interest] at [location] is [value for current and future are ...], [according to the Wikipedia article] the required [parameter] for [object of interest] is [value]. [Two sentence of clarification, with criitcal montly-based assessment of the potential changes]'
    'Repeat for each parameter.'
    </Important>
    """
    
    
    #[1] Tool description for netCDF extraction
    class get_data_components_args(BaseModel):
        environmental_data: Optional[Union[str, Literal["Temperature", "Precipitation", "u_wind", "v_wind"]]] = Field(
            default=None,
            description="The type of environmental data to retrieve. Choose from Temperature, Precipitation, u_wind, or v_wind.",
            enum_description={
                "Temperature": "The mean monthly temperature data.",
                "Precipitation": "The mean monthly precipitation data.",
                "u_wind": "The mean monthly u wind component data.",
                "v_wind": "The mean monthly v wind component data."
            }
        )
        months: Optional[Union[str, List[Literal["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]]] = Field(
            default=None,
            description="List of months or a stringified list of month names to retrieve data for. Each month should be one of 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'. If not specified, data for all months will be retrieved."
        )

    def get_data_components(**kwargs):
        stream_handler.update_progress("Retrieving data for advanced analysis with a smart agent...")

        if isinstance(kwargs.get("months"), str):
            try:
                kwargs["months"] = ast.literal_eval(kwargs["months"])
            except (ValueError, SyntaxError):
                # Optional: handle invalid input
                kwargs["months"] = None  # or raise an exception  
        args = get_data_components_args(**kwargs)
        # Parse the arguments using the args_schema
        environmental_data = args.environmental_data
        months = args.months  # List of month names
        if environmental_data is None:
            return {"error": "No environmental data type specified."}
        if environmental_data not in ["Temperature", "Precipitation", "u_wind", "v_wind"]:
            return {"error": f"Invalid environmental data type: {environmental_data}"}
       
        if config['use_high_resolution_climate_model']:
            df_list = state.df_list
            response = {}                 

            environmental_mapping = {
                "Temperature": "mean2t",
                "Precipitation": "tp",
                "u_wind": "wind_u",
                "v_wind": "wind_v"
            }

            if environmental_data not in environmental_mapping:
                return {"error": f"Invalid environmental data type: {environmental_data}"}
            
            # Filter the DataFrame for the selected months and extract the values
            var_name = environmental_mapping[environmental_data]
            
            if not months:
                months = [calendar.month_abbr[m] for m in range(1, 13)]
                
            # Create a mapping from abbreviated to full month names
            month_mapping = {calendar.month_abbr[m]: calendar.month_name[m] for m in range(1, 13)}
            selected_months = [month_mapping[abbr] for abbr in months]

            for entry in df_list:
                df = entry.get('dataframe')
                var_meta = entry.get('extracted_vars').get(var_name)
                if df is None:
                    raise ValueError(f"Entry does not contain a 'dataframe' key.")
                    
                data_values = df[df['Month'].isin(selected_months)][var_name].tolist()
                ext_data = {month: np.round(value,2) for month, value in zip(selected_months, data_values)}
                ext_exp = f"Monthly mean values of {environmental_data}, {var_meta['units']} for years: " +entry['years_of_averaging']
                response.update({ext_exp: ext_data})
            return response    
        else: #config['use_high_resolution_climate_model']
            lat = float(state.input_params['lat'])
            lon = float(state.input_params['lon'])
            data_path = config['data_settings']['data_path']

            # Dictionaries for historical and SSP585 data files
            data_files_historical = {
                "Temperature": ("AWI_CM_mm_historical.nc", "tas"),
                "Precipitation": ("AWI_CM_mm_historical_pr.nc", "pr"),
                "u_wind": ("AWI_CM_mm_historical_uas.nc", "uas"),
                "v_wind": ("AWI_CM_mm_historical_vas.nc", "vas")
            }

            data_files_ssp585 = {
                "Temperature": ("AWI_CM_mm_ssp585.nc", "tas"),
                "Precipitation": ("AWI_CM_mm_ssp585_pr.nc", "pr"),
                "u_wind": ("AWI_CM_mm_ssp585_uas.nc", "uas"),
                "v_wind": ("AWI_CM_mm_ssp585_vas.nc", "vas")
            }

            if environmental_data not in data_files_historical:
                return {"error": f"Invalid environmental data type: {environmental_data}"}

            # Get file names and variable names for both datasets
            file_name_hist, var_name_hist = data_files_historical[environmental_data]
            file_name_ssp585, var_name_ssp585 = data_files_ssp585[environmental_data]

            # Build file paths
            file_path_hist = os.path.join(data_path, file_name_hist)
            file_path_ssp585 = os.path.join(data_path, file_name_ssp585)

            # Check if files exist
            if not os.path.exists(file_path_hist):
                return {"error": f"Data file {file_name_hist} not found in {data_path}"}
            if not os.path.exists(file_path_ssp585):
                return {"error": f"Data file {file_name_ssp585} not found in {data_path}"}

            # Open datasets
            dataset_hist = nc.Dataset(file_path_hist)
            dataset_ssp585 = nc.Dataset(file_path_ssp585)

            # Get latitude and longitude arrays
            lats_hist = dataset_hist.variables['lat'][:]
            lons_hist = dataset_hist.variables['lon'][:]
            lats_ssp585 = dataset_ssp585.variables['lat'][:]
            lons_ssp585 = dataset_ssp585.variables['lon'][:]

            # Find the nearest indices for historical data
            lat_idx_hist = (np.abs(lats_hist - lat)).argmin()
            lon_idx_hist = (np.abs(lons_hist - lon)).argmin()

            # Find the nearest indices for SSP585 data
            lat_idx_ssp585 = (np.abs(lats_ssp585 - lat)).argmin()
            lon_idx_ssp585 = (np.abs(lons_ssp585 - lon)).argmin()

            # Extract data at the specified location
            data_hist = dataset_hist.variables[var_name_hist][:, :, :, lat_idx_hist, lon_idx_hist]
            data_ssp585 = dataset_ssp585.variables[var_name_ssp585][:, :, :, lat_idx_ssp585, lon_idx_ssp585]

            # Squeeze data to remove singleton dimensions (shape becomes (12,))
            data_hist = np.squeeze(data_hist)
            data_ssp585 = np.squeeze(data_ssp585)

            # Process data according to the variable
            if environmental_data == "Temperature":
                # Convert from Kelvin to Celsius
                data_hist = data_hist - 273.15
                data_ssp585 = data_ssp585 - 273.15
                units = "°C"
            elif environmental_data == "Precipitation":
                # Convert from kg m-2 s-1 to mm/month
                days_in_month = np.array([31, 28, 31, 30, 31, 30,
                                        31, 31, 30, 31, 30, 31])
                seconds_in_month = days_in_month * 24 * 3600  # seconds in each month
                data_hist = data_hist * seconds_in_month
                data_ssp585 = data_ssp585 * seconds_in_month
                units = "mm/month"
            elif environmental_data in ["u_wind", "v_wind"]:
                # Units are already in m/s
                units = "m/s"
            else:
                units = "unknown"

            # Close datasets
            dataset_hist.close()
            dataset_ssp585.close()

            # List of all month names
            all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Map month names to indices
            month_indices = {month: idx for idx, month in enumerate(all_months)}

            # If months are specified, select data for those months
            if months:
                # Validate months
                valid_months = [month for month in months if month in month_indices]
                if not valid_months:
                    return {"error": "Invalid months provided."}
                selected_indices = [month_indices[month] for month in valid_months]
                selected_months = valid_months
            else:
                # Use all months if none are specified
                selected_indices = list(range(12))
                selected_months = all_months

            # Subset data for selected months
            data_hist = data_hist[selected_indices]
            data_ssp585 = data_ssp585[selected_indices]

            # Create dictionaries mapping months to values with units
            hist_data_dict = {month: f"{value:.2f} {units}" for month, value in zip(selected_months, data_hist)}
            ssp585_data_dict = {month: f"{value:.2f} {units}" for month, value in zip(selected_months, data_ssp585)}

            # Return both historical and SSP585 data
            return {
                f"{environmental_data}_historical": hist_data_dict,
                f"{environmental_data}_ssp585": ssp585_data_dict
            }

    # Define the data_extraction_tool
    data_extraction_tool = StructuredTool.from_function(
        func=get_data_components,
        name="get_data_components",
        description="Retrieve the necessary environmental data component.",
        args_schema=get_data_components_args
    )

    #[2] Wikipedia processing tool
    def process_wikipedia_article(query: str) -> str:     
        stream_handler.update_progress("Searching Wikipedia for related information with a smart agent...")
   
        # Initialize the LLM
        if config['model_type'] == "local":
            llm = ChatOpenAI(
                openai_api_base="http://localhost:8000/v1",
                model_name=config['model_name_agents'],  # Match the exact model name you used
                openai_api_key=api_key_local,
                temperature  = temperature,
            )                          
        elif config['model_type'] == "openai":
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=config['model_name_tools'],
                temperature=temperature
            )
        elif config['model_type'] == "aitta":
            llm = get_aitta_chat_model(
                config['model_name_tools'], temperature = temperature)
        # Define your custom prompt template
        template = """
        Read the provided  {wikipage} carefully. Extract and present information related to the following keywords relative to {question}:

            • Temperature
            • Precipitation
            • Wind
            • Elevation Above Sea Level
            • Population
            • Natural Hazards
            • Soil Type

        Guidelines:

            • For each keyword, if information is available in the article, extract and present both qualitative and quantitative information separately.
            • Separate the information into two sections: “Qualitative” and “Quantitative” for each keyword.
            • In the Quantitative section, focus on specific numerical data, measurements, thresholds, or quantitative values associated with the keyword. Include units (e.g., °C, mm, km/h) where applicable.
            • In the Qualitative section, provide relevant descriptive information that does not include specific numbers.
            • If no information is available for a keyword, omit it entirely—do not mention the keyword.
            • Present the information in a clear and organized manner.
            • Do not include any additional information beyond what is relevant to the listed keywords.

        Example Format:

            • Temperature:
                • Quantitative: Requires warm days above 10°C (50°F) for flowering.
                • Qualitative: Maize is cold-intolerant and must be planted in the spring in temperate zones.
            • Wind:
                • Quantitative: Can be uprooted by winds exceeding 60km/h due to shallow roots.
                • Qualitative: Maize pollen is dispersed by wind.
            • Soil Type:
                • Quantitative: Prefers soils with a pH between 6.0-7.5.
                • Qualitative: Maize is intolerant of nutrient-deficient soils and depends on adequate soil moisture.

        Note: Replace the placeholders with the actual qualitative and quantitative information extracted from the article, ensuring that each piece of information is placed in the appropriate section.
        """
        
        #class EncyclopediaLoader:
        #    def __init__(self, topic):
        #        self.topic = topic

        #    def load(self):
        #        # Encode the topic to be URL-friendly
        #        encoded_topic = quote_plus(self.topic)
        #        url = f'https://encyclopedia2.thefreedictionary.com/{encoded_topic}'
                
        #        # Fetch the page
        #        response = requests.get(url)
        #        if response.status_code != 200:
        #            raise Exception(f"Failed to retrieve article for topic: {self.topic}")
                
        #        # Parse the HTML content
        #        soup = BeautifulSoup(response.content, 'html.parser')
                
        #        # Find the div containing the article
        #        article_div = soup.find('div', {'id': 'Definition'})
        #        if not article_div:
        #            raise Exception(f"Article content not found for topic: {self.topic}")
                
        #        # Extract text from all paragraphs within the article div
        #        paragraphs = article_div.find_all('p')
        #        article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
        #        document = Document(page_content=article_text, metadata={"source": url})

                
        #        return document 

        #data_rag = config['rag_settings']['data_path']

        # Create the prompt
        prompt = ChatPromptTemplate.from_template(template)

        # Load the Wikipedia article
        loader = WikipediaLoader(
            query=query,
            load_all_available_meta=True,
            doc_content_chars_max=100000,
            load_max_docs=1
        )
        raw_documents = loader.load()

        #try:
        #    loader = EncyclopediaLoader(query)
        #    article_text = loader.load().page_content
        #except Exception as e:
        #    print(f"Encyclopedia loader failed: {str(e)}")
        #    article_text = ""  # Set empty string if encyclopedia lookup fails

        #raw_documents.append(article_text) 
        #content_str = 'Encyclopedia article: ' + article_text + '\n' + 'Wikipedia article: ' + raw_documents[0].page_content
        content_str = 'Wikipedia article: ' + raw_documents[0].page_content
        if not raw_documents:
            return "No Wikipedia article found for the query."

        # Run the chain
        chain = prompt | llm
        result = chain.invoke({"wikipage": content_str, "question": query})
        title = raw_documents[0].metadata.get("title", "").replace(" ", "_")
        ref = "Wikipedia URL: " + f"https://en.wikipedia.org/wiki/{title}"
        return {
            "result": result,
            "references": ref
                }

    # Define the args schema for the Wikipedia tool
    class WikipediaSearchArgs(BaseModel):
        query: str = Field(
            description="The topic to search on Wikipedia."
        )

    # Create the Wikipedia tool
    wikipedia_tool = StructuredTool.from_function(
        func=process_wikipedia_article,
        name="wikipedia_search",
        description=(
            "A tool to search Wikipedia for information and process it according to specific guidelines. "
            "Input should be the topic of interest. For example, if the question is about growing corn in southern Germany, "
            "you should input 'corn' as the query."
        ),
        args_schema=WikipediaSearchArgs
    )

    #[3] RAG extraction tool
    class RAGSearchArgs(BaseModel):
        query: str = Field(
            description="The topic to search in the internal knowledge database."
        )
        #domain: str = Field(
        #    description="The domain of the internal knowledge database to search in. For example 'agriculture', 'energy', 'water', 'health', etc."
        #)


    def process_RAG_search(query: str) -> str:
        # Retrieve the path to the vector store
        data_rag = config['rag_articles']['data_path']
        
        # Load the persisted vector store
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = Chroma(
            persist_directory=data_rag,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever()

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs:
            return "No relevant documents found for the query."
        
        # Combine the documents into a single string
        rag_content = '\n\n'.join([doc.page_content for doc in retrieved_docs])
        rag_references = [doc.metadata.get("source", "") for doc in retrieved_docs]
        
            # Define your custom prompt template
        template = """
        Read the provided {rag_content} carefully. Extract and present information related to the following keywords relative to {question}:

            • Temperature
            • Precipitation
            • Wind
            • Elevation Above Sea Level
            • Population
            • Natural Hazards
            • Soil Type

        Guidelines:

            • For each keyword, if information is available in the article, extract and present both qualitative and quantitative information separately.
            • Separate the information into two sections: “Qualitative” and “Quantitative” for each keyword.
            • In the Quantitative section, focus on specific numerical data, measurements, thresholds, or quantitative values associated with the keyword. Include units (e.g., °C, mm, km/h) where applicable.
            • In the Qualitative section, provide relevant descriptive information that does not include specific numbers.
            • If no information is available for a keyword, omit it entirely—do not mention the keyword.
            • Present the information in a clear and organized manner.
            • Do not include any additional information beyond what is relevant to the listed keywords.

        Example Format:

            • Temperature:
                • Quantitative: Requires warm days above 10 °C (50 °F) for flowering.
                • Qualitative: Maize is cold-intolerant and must be planted in the spring in temperate zones.
            • Wind:
                • Quantitative: Can be uprooted by winds exceeding 60 km/h due to shallow roots.
                • Qualitative: Maize pollen is dispersed by wind.
            • Soil Type:
                • Quantitative: Prefers soils with a pH between 6.0-7.5.
                • Qualitative: Maize is intolerant of nutrient-deficient soils and depends on adequate soil moisture.

        Note: Replace the placeholders with the actual qualitative and quantitative information extracted from the article, ensuring that each piece of information is placed in the appropriate section.
        <Important> DO NOT include schema if no relevant information is found. Just return 'No relevant information found for the query.' AND NOTHING ELSE. </Important>
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize the LLM
        if config['model_type'] == "local":
            llm = ChatOpenAI(
                openai_api_base="http://localhost:8000/v1",
                model_name=config['model_name_tools'],  # Match the exact model name you used
                openai_api_key=api_key_local,
                temperature  = temperature,
            )                          
        elif config['model_type'] == "openai":
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=config['model_name_tools'],
                temperature=temperature
            )        
        elif config['model_type'] == "aitta":
            llm = get_aitta_chat_model(
                config['model_name_tools'], temperature = temperature)
        
        # Create the chain with the prompt and LLM
        chain = prompt | llm
        
        # Run the chain with the provided content and question
        result = chain.invoke({"rag_content": rag_content, "question": query})
        
        return {
            "result": result,
            "references": rag_references
                }


    # Create the RAG tool
    rag_tool = StructuredTool.from_function(
        func=process_RAG_search,
        name="RAG_search",
        description="A tool to answer questions about environmental conditions related to user question. For search queries, provide expanded question. For example, if the question is about growing corn, the query should be 'limiting factors for corn growth'.",
        args_schema=RAGSearchArgs
    )
    # [4] ECOCROP tool 
    class EcoCropSearchArgs(BaseModel):
        query: str = Field(
            description="The name of the crop to search for in the ECOCROP database."
        )
    def process_ecocrop_search(query: str) -> str:

        stream_handler.update_progress("Searching ECOCROP for related information with a smart agent...")
        # Load the ECOCROP database
        ecocroploc = config['ecocrop']['ecocroploc_path']
        variable_expl_path = config['ecocrop']['variable_expl_path']
        rag_ecocrop = config['ecocrop']['data_path']
        if not os.path.exists(ecocroploc) or not os.path.exists(variable_expl_path):
            return f"ECOCROP database files not found."

        ecocropall = pd.read_csv(ecocroploc, encoding='latin1', engine="python")
        variable_expl = pd.read_csv(variable_expl_path, engine='python')
        variable_dict = dict(zip(variable_expl['Variable'], variable_expl['Explanation']))

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=rag_ecocrop
        )

        # Initialize the LLM
        if config['model_type'] == "local":
            llm = ChatOpenAI(
                openai_api_base="http://localhost:8000/v1",
                model_name=config['model_name_tools'],  # Match the exact model name you used
                openai_api_key=api_key_local,
                temperature  = 0,
            )                  
        elif config['model_type'] == "openai":        
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=config['model_name_tools'],
                temperature=0.0
            )        
        elif config['model_type'] == "aitta":
            llm = get_aitta_chat_model(config['model_name_tools'], temperature = 0)

        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        Your task is to select the most relevant document for the following query:
        Query: {query}
        Here are the documents to choose from:
        {documents}

        Please analyze these documents and provide ONLY the SOURCE of the single most relevant one that best matches the query.
        DO NOT add anything else in your response except the SOURCE of the document.
        """)

        # Perform the similarity search
        query_result = vector_store.similarity_search(query=query, k=10)
        if not query_result:
            return f"No data found for {query}."

        # Extract the document sources
        messages = prompt.format_messages(query=query, documents=query_result)
        response = llm.invoke(messages)
    
        # Filter the data
        filtered_data = ecocropall[ecocropall['ScientificName'] == response.content.strip()]
        if filtered_data.empty:
            return f"No data found for {query}."

        # Format the output
        ecocrop_output = f"Data from ECOCROP database for {query}:\n"
        ecocrop_output += f"Scientific Name: {filtered_data['ScientificName'].iloc[0]}\n"
        for varname in variable_dict.keys():
            value = filtered_data[varname].iloc[0]
            if pd.notna(value):
                ecocrop_output += f"{variable_dict[varname]}: {value}\n"

        return ecocrop_output
        # Create the ECOCROP tool
    ecocrop_tool = StructuredTool.from_function(
        func=process_ecocrop_search,
        name="ECOCROP_search",
        description=(
            "A tool to retrieve crop-specific environmental requirements from the ECOCROP database. "
            "Input should be the name of the crop."
        ),
        args_schema=EcoCropSearchArgs
    )
    # Initialize the LLM
    if config['model_type'] == "local":
        llm = ChatOpenAI(
            openai_api_base="http://localhost:8000/v1",
            model_name=config['model_name_agents'],  # Match the exact model name you used
            openai_api_key=api_key_local,
            temperature  = 0,
        )                  
    elif config['model_type'] == "openai":        
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config['model_name_agents'],
            temperature=0.0
        )
    elif config['model_type'] == "aitta":
        llm = get_aitta_chat_model(config['model_name_tools'], temperature = 0)

    # List of tools
    tools = [data_extraction_tool, rag_tool,wikipedia_tool, ecocrop_tool]

    # Create the agent with the tools and prompt
    prompt += """\nadditional information:\n
    question is related to this location: {location_str} \n
    """
    
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("user", state.user),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True  # Set to True to capture tool outputs
    )

    # Prepare input for the agent
    agent_input = {
        "input": state.user,
        "chat_history": state.messages,
        "lat": lat,
        "lon": lon,
        "location_str": state.input_params['location_str']
    }


    # Run the agent
    result = agent_executor(agent_input)

    # Extract the tool outputs
    tool_outputs = {}
    for action, observation in result['intermediate_steps']:
        if action.tool == 'wikipedia_search':
            if isinstance(observation, AIMessage):
                tool_outputs['wikipedia_search'] = observation.content
            else:
                output = observation
            # Assuming output_data is a dict now
            if isinstance(output, dict):
                tool_outputs['wikipedia_search'] = output.get('result').content
                state.references.append(output.get('references', []))
            else:
                tool_outputs['wikipedia_search'] = output
        elif action.tool == 'get_data_components':
            if isinstance(observation, AIMessage):
                tool_outputs['get_data_components'] = observation.content
            else:
                tool_outputs['get_data_components'] = observation
        elif action.tool == 'ECOCROP_search':
            if isinstance(observation, AIMessage):
                tool_outputs['ECOCROP_search'] = observation.content
            else:
                tool_outputs['ECOCROP_search'] = observation
            if any("FAO, IIASA" not in element for element in state.references):    
                state.references.append("FAO, IIASA: Global Agro-Ecological Zones (GAEZ V4) - Data Portal User's Guide, 1st edn. FAO and IIASA, Rome, Italy (2021). https://doi.org/10.4060/cb5167en")    
        if action.tool == 'RAG_search':
            if isinstance(observation, AIMessage):
                tool_outputs['RAG_search'] = observation.content
            else:
                output = observation
            # Assuming output_data is a dict now
            if isinstance(output, dict):
                tool_outputs['RAG_search'] = output.get('result').content
                # If refs is a list, extend; if it's a string, append.
                refs = output.get('references', [])
                if isinstance(refs, list):
                    state.references.extend(refs)
                elif isinstance(refs, str):
                    state.references.append(refs)                

    # Store the response from the wikipedia_search tool into state
    if 'wikipedia_search' in tool_outputs:
        state.wikipedia_tool_response = tool_outputs['wikipedia_search']
    if 'ECOCROP_search' in tool_outputs:
        state.ecocrop_search_response = tool_outputs['ECOCROP_search']
    if 'RAG_search' in tool_outputs:
        state.rag_search_response = tool_outputs['RAG_search']
        
    # Also store the agent's final answer
    smart_agent_response = result['output']
    state.smart_agent_response = {'output': smart_agent_response}
    # Return both smart_agent_response and wikipedia_tool_response
    return {
        'smart_agent_response': state.smart_agent_response,
        'wikipedia_tool_response': state.wikipedia_tool_response,
        'ecocrop_search_response': state.ecocrop_search_response,
        'rag_search_response': state.rag_search_response,
        'references': state.references
    }
