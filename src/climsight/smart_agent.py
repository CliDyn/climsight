# smart_agent.py

from pydantic import BaseModel, Field
from typing import Optional, Literal
import netCDF4 as nc
import numpy as np
import os

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
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from langchain.schema import Document

# Import AgentState from climsight_classes
from climsight_classes import AgentState

def smart_agent(state: AgentState, config, api_key):

    lat = float(state.input_params['lat'])
    lon = float(state.input_params['lon'])

    # System prompt
    prompt = f"""
    You are the smart agent of ClimSight. Your task is to retrieve necessary components of the climatic datasets based on the user's request.
    You have access to tools called "get_data_components", "wikipedia_search", and "RAG_search" which you can use to retrieve the necessary environmental data components.
    - "get_data_components" will retrieve the necessary data from the climatic datasets at the location of interest (latitude: {lat}, longitude: {lon}).
    - "wikipedia_search" will help you determine the necessary data to retrieve with the get_data_components tool.
    - "RAG_search" can provide detailed information about environmental conditions for growing corn from your internal knowledge base.
    <Important> ALWAYS call FIRST SIMULTANIOUSLY the wikipedia_search and RAG_search; it will help you determine the necessary data to retrieve with the get_data_components tool. At second step, call the get_data_components tool with the necessary data.</Important>
    Use these tools to get the data you need to answer the user's question.
    After retrieving the data, provide a concise summary of the parameters you retrieved, explaining briefly why they are important. Keep your response short and to the point.
    Do not include any additional explanations or reasoning beyond the concise summary.
    Do not include any chain-of-thought reasoning or action steps in your final answer.

    <Important> 
    For the final response try to follow the following format:
    'The [retrieved values of the parameter] for the [object of interest] at [location] is [value], [according to the Wikipedia article] the required [parameter] [value] for [object of interest] is [value]. [One sentence of clarification]'
    'Repeat for each parameter.'
    </Important>
    """

    #[1] Tool description for netCDF extraction
    class get_data_components_args(BaseModel):
        environmental_data: Optional[Literal["Temperature", "Precipitation", "u_wind", "v_wind"]] = Field(
            default=None,
            description="The type of environmental data to retrieve. Choose from Temperature, Precipitation, u_wind, or v_wind.",
            enum_description={
                "Temperature": "The mean monthly temperature data.",
                "Precipitation": "The mean monthly precipitation data.",
                "u_wind": "The mean monthly u wind component data.",
                "v_wind": "The mean monthly v wind component data."
            }
        )

    def get_data_components(**kwargs):
        # Parse the arguments using the args_schema
        args = get_data_components_args(**kwargs)
        environmental_data = args.environmental_data
        if environmental_data is None:
            return {"error": "No environmental data type specified."}

        lat = float(state.input_params['lat'])
        lon = float(state.input_params['lon'])
        data_path = config['data_settings']['data_path']

        data_files = {
            "Temperature": ("AWI_CM_mm_historical.nc", "tas"),
            "Precipitation": ("AWI_CM_mm_historical_pr.nc", "pr"),
            "u_wind": ("AWI_CM_mm_historical_uas.nc", "uas"),
            "v_wind": ("AWI_CM_mm_historical_vas.nc", "vas")
        }

        if environmental_data not in data_files:
            return {"error": f"Invalid environmental data type: {environmental_data}"}

        file_name, var_name = data_files[environmental_data]
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            return {"error": f"Data file {file_name} not found in {data_path}"}

        dataset = nc.Dataset(file_path)
        lats = dataset.variables['lat'][:]
        lons = dataset.variables['lon'][:]
        # Find the nearest indices
        lat_idx = (np.abs(lats - lat)).argmin()
        lon_idx = (np.abs(lons - lon)).argmin()
        data = dataset.variables[var_name][:, :, :, lat_idx, lon_idx]
        # Take mean over all axes to get a single value
        point_data = np.mean(data)
        dataset.close()

        return {environmental_data: point_data}

    # Define the data_extraction_tool
    data_extraction_tool = StructuredTool.from_function(
        func=get_data_components,
        name="get_data_components",
        description="Retrieve the necessary environmental data component.",
        args_schema=get_data_components_args
    )

    #[2] Wikipedia processing tool
    def process_wikipedia_article(query: str) -> str:
        from langchain.document_loaders import WikipediaLoader
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        # Initialize the LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config['model_name'],
            temperature=0.0
        )

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
                • Quantitative: Requires warm days above 10 °C (50 °F) for flowering.
                • Qualitative: Maize is cold-intolerant and must be planted in the spring in temperate zones.
            • Wind:
                • Quantitative: Can be uprooted by winds exceeding 60 km/h due to shallow roots.
                • Qualitative: Maize pollen is dispersed by wind.
            • Soil Type:
                • Quantitative: Prefers soils with a pH between 6.0–7.5.
                • Qualitative: Maize is intolerant of nutrient-deficient soils and depends on adequate soil moisture.

        Note: Replace the placeholders with the actual qualitative and quantitative information extracted from the article, ensuring that each piece of information is placed in the appropriate section.
        """
        
        class EncyclopediaLoader:
            def __init__(self, topic):
                self.topic = topic

            def load(self):
                # Encode the topic to be URL-friendly
                encoded_topic = quote_plus(self.topic)
                url = f'https://encyclopedia2.thefreedictionary.com/{encoded_topic}'
                
                # Fetch the page
                response = requests.get(url)
                if response.status_code != 200:
                    raise Exception(f"Failed to retrieve article for topic: {self.topic}")
                
                # Parse the HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the div containing the article
                article_div = soup.find('div', {'id': 'Definition'})
                if not article_div:
                    raise Exception(f"Article content not found for topic: {self.topic}")
                
                # Extract text from all paragraphs within the article div
                paragraphs = article_div.find_all('p')
                article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                document = Document(page_content=article_text, metadata={"source": url})

                
                return document 

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

        try:
            loader = EncyclopediaLoader(query)
            article_text = loader.load().page_content
        except Exception as e:
            print(f"Encyclopedia loader failed: {str(e)}")
            article_text = ""  # Set empty string if encyclopedia lookup fails

        #raw_documents.append(article_text) 
        content_str = 'Encyclopedia article: ' + article_text + '\n' + 'Wikipedia article: ' + raw_documents[0].page_content

        if not raw_documents:
            return "No Wikipedia article found for the query."

        # Run the chain
        chain = prompt | llm
        result = chain.invoke({"wikipage": content_str, "question": query})

        return result

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
                • Quantitative: Requires warm days above 10 °C (50 °F) for flowering.
                • Qualitative: Maize is cold-intolerant and must be planted in the spring in temperate zones.
            • Wind:
                • Quantitative: Can be uprooted by winds exceeding 60 km/h due to shallow roots.
                • Qualitative: Maize pollen is dispersed by wind.
            • Soil Type:
                • Quantitative: Prefers soils with a pH between 6.0–7.5.
                • Qualitative: Maize is intolerant of nutrient-deficient soils and depends on adequate soil moisture.

        Note: Replace the placeholders with the actual qualitative and quantitative information extracted from the article, ensuring that each piece of information is placed in the appropriate section.
        <Important> DO NOT include schema if no relevant information is found. Just return 'No relevant information found for the query.' AND NOTHING ELSE. </Important>
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize the LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config['model_name'],
            temperature=0.0
        )
        
        # Create the chain with the prompt and LLM
        chain = prompt | llm
        
        # Run the chain with the provided content and question
        result = chain.invoke({"rag_content": rag_content, "question": query})
        
        return result


    # Create the RAG tool
    rag_tool = StructuredTool.from_function(
        func=process_RAG_search,
        name="RAG_search",
        description="A tool to answer questions about environmental conditions for growing corn. For search queries, provide expanded question. For example, if the question is about growing corn, the query should be 'limiting factors for corn growth'.",
        args_schema=RAGSearchArgs
    )

    # Initialize the LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=config['model_name'],
        temperature=0.0
    )

    # List of tools
    tools = [data_extraction_tool, wikipedia_tool, rag_tool]

    # Create the agent with the tools and prompt
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
        "lon": lon
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
                tool_outputs['wikipedia_search'] = observation
        elif action.tool == 'get_data_components':
            if isinstance(observation, AIMessage):
                tool_outputs['get_data_components'] = observation.content
            else:
                tool_outputs['get_data_components'] = observation

    # Store the response from the wikipedia_search tool into state
    if 'wikipedia_search' in tool_outputs:
        state.wikipedia_tool_response = tool_outputs['wikipedia_search']

    # Also store the agent's final answer
    smart_agent_response = result['output']
    state.smart_agent_response = {'output': smart_agent_response}

    # Return both smart_agent_response and wikipedia_tool_response
    return {
        'smart_agent_response': state.smart_agent_response,
        'wikipedia_tool_response': state.wikipedia_tool_response
    }