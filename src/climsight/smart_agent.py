# smart_agent.py

from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import netCDF4 as nc
import numpy as np
import os
import ast
from typing import Union

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, Tool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

#Import tools
from tools.python_repl import create_python_repl_tool
from tools.image_viewer import create_image_viewer_tool
# era5_retrieval_tool is used only in data_analysis_agent

#import requests
#from bs4 import BeautifulSoup
#from urllib.parse import quote_plus
#from langchain_core.documents import Document


#Import for working Path
import uuid
import streamlit as st
from pathlib import Path
try:
    from aitta_client import Model, Client
    from aitta_client.authentication import APIKeyAccessTokenSource
except:
    pass

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
    if "o1" in config['llm_smart']['model_name']:
        temperature = 1


    if 'session_uuid' not in st.session_state:
        st.session_state.session_uuid = str(uuid.uuid4())

    # Create working directory
    work_dir = Path("tmp/sandbox") / st.session_state.session_uuid
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir_str = str(work_dir.resolve())

    # System prompt
    prompt = f"""
    You are the information gathering agent of ClimSight. Your task is to collect relevant background
    information about the user's query using external knowledge sources.

    You have access to three information retrieval tools:
    - "wikipedia_search": Search Wikipedia for general information about the topic of interest.
      Use this to understand the broader context of the user's question.

    - "RAG_search": Query ClimSight's internal climate knowledge base for scientific details.
      Use this to find detailed scientific information from climate literature and research.

    - "ECOCROP_search": Get crop-specific environmental requirements from the ECOCROP database.
      <Important> ONLY use this tool if the user's question is clearly about agriculture or crops.
      Do NOT use it for general climate queries. </Important>

    - "python_repl" allows you to execute Python code for data analysis, visualization, and calculations.
         <Important> Use this tool when:
        - Creating visualizations (plots, charts, graphs) of climate data

        **CRITICAL: Your working directory is available at `work_dir` = '{work_dir_str}'**
        **When saving plots, ALWAYS store the full path in a variable for later use!**

        **CORRECT way to save and reference images:**
           ```python
           plot_path = f'{{{{work_dir}}}}/temperature_plot.png'
           plt.savefig(plot_path)
           print(plot_path) # PRINT IT TO CONFIRM
           ```
        </Important>

    **Your goal**: Gather comprehensive background information and compile it into a well-structured
    summary that will be used by subsequent agents for data analysis.

    **Important guidelines**:
    - When citing information from Wikipedia or RAG sources, include source references in your summary
    - Present ECOCROP database results exactly as they are provided (no citations needed for database facts)
    - Focus on gathering contextual information, NOT on extracting or analyzing specific climate data
    - Do NOT attempt to retrieve climate data values - that will be handled by other agents
    - Your output should be a comprehensive text summary with relevant background context
    - If you call a tool multiple times, incorporate all results in your summary
    - Do NOT call wikipedia_search more than 10 times total
    """
    if config['llm_smart']['model_type'] in ("local", "aitta"):
        prompt += f"""

        <Important> - Tool use order. Call the wikipedia_search, RAG_search, and ECOCROP_search tools as needed,
        but only one at a time per turn. Gather all relevant background information. </Important>
        """
    else:
        prompt += f"""

        <Important> - Tool use order. Call wikipedia_search, RAG_search, and ECOCROP_search
        (if applicable) SIMULTANEOUSLY to gather background information efficiently. </Important>

        """
    prompt += f"""

    **Output format**:
    After gathering information from the available tools, provide a well-structured summary organized as follows:

    1. **General Background** (from Wikipedia if available):
       - Key facts and context about the topic
       - Relevant qualitative and quantitative information
       - Include specific values with units where mentioned

    2. **Scientific Context** (from RAG if available):
       - Detailed scientific information from climate literature
       - Relevant research findings and climate insights
       - Technical details that provide context

    3. **Specific Requirements** (from ECOCROP if applicable):
       - Database results presented exactly as provided
       - Environmental thresholds and requirements
       - Optimal and tolerable ranges

    4. **Key Takeaways**:
       - Summarize the most important points for understanding the user's query
       - Highlight critical factors or thresholds mentioned

    <Important>
    - Keep your summary concise but comprehensive
    - Do NOT include chain-of-thought reasoning or tool invocation details
    - Do NOT mention what you're going to do or explain your process
    - Focus on presenting the gathered information in a clear, organized format
    - The summary should help subsequent agents understand the context for data analysis
    - If you have already called wikipedia_search 10 times, proceed without further Wikipedia calls
    </Important>
    """

    #[1] get_data_components tool moved to data_analysis_agent.py

    #[2] Wikipedia processing tool
    wikipedia_call_state = {"count": 0}

    def process_wikipedia_article(query: str) -> str:
        if wikipedia_call_state["count"] >= 10:
            return ""
        wikipedia_call_state["count"] += 1

        stream_handler.update_progress("Searching Wikipedia for related information with a smart agent...")
   
        # Initialize the LLM
        if config['llm_smart']['model_type'] == "local":
            llm = ChatOpenAI(
                openai_api_base="http://localhost:8000/v1",
                model_name=config['llm_smart']['model_name'],  # Match the exact model name you used
                openai_api_key=api_key_local,
                temperature  = temperature,
            )                          
        elif config['llm_smart']['model_type'] == "openai":
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=config['llm_smart']['model_name'],
                temperature=temperature
            )
        elif config['llm_smart']['model_type'] == "aitta":
            llm = get_aitta_chat_model(
                config['llm_smart']['model_name'], temperature = temperature)
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
        # Use invoke() for newer LangChain versions, fallback to get_relevant_documents() for older
        try:
            retrieved_docs = retriever.invoke(query)
        except AttributeError:
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
        if config['llm_smart']['model_type'] == "local":
            llm = ChatOpenAI(
                openai_api_base="http://localhost:8000/v1",
                model_name=config['llm_smart']['model_name'],  # Match the exact model name you used
                openai_api_key=api_key_local,
                temperature  = temperature,
            )                          
        elif config['llm_smart']['model_type'] == "openai":
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=config['llm_smart']['model_name'],
                temperature=temperature
            )        
        elif config['llm_smart']['model_type'] == "aitta":
            llm = get_aitta_chat_model(
                config['llm_smart']['model_name'], temperature = temperature)
        
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
        if config['llm_smart']['model_type'] == "local":
            llm = ChatOpenAI(
                openai_api_base="http://localhost:8000/v1",
                model_name=config['llm_smart']['model_name'],  # Match the exact model name you used
                openai_api_key=api_key_local,
                temperature  = 0,
            )                  
        elif config['llm_smart']['model_type'] == "openai":        
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=config['llm_smart']['model_name'],
                temperature=0.0
            )        
        elif config['llm_smart']['model_type'] == "aitta":
            llm = get_aitta_chat_model(config['llm_smart']['model_name'], temperature = 0)

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


    # Create python_repl tool
    python_repl_tool = create_python_repl_tool()

    # Initialize the LLM
    if config['llm_smart']['model_type'] == "local":
        llm = ChatOpenAI(
            openai_api_base="http://localhost:8000/v1",
            model_name=config['llm_smart']['model_name'],
            openai_api_key=api_key_local,
            temperature  = 0,
        )                  
    elif config['llm_smart']['model_type'] == "openai":        
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config['llm_smart']['model_name'],
            temperature=0.0
        )

    elif config['llm_smart']['model_type'] == "aitta":
        llm = get_aitta_chat_model(config['llm_smart']['model_name'], temperature = 0)

    # List of tools
    tools = [rag_tool, ecocrop_tool, wikipedia_tool, python_repl_tool]

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
    wikipedia_results = []
    rag_results = []

    def _extract_tool_query(action):
        tool_input = getattr(action, "tool_input", None)
        if tool_input is None:
            tool_input = getattr(action, "input", None)
        if isinstance(tool_input, dict):
            return tool_input.get("query") or tool_input.get("input") or tool_input
        return tool_input

    def _normalize_tool_output(observation):
        if isinstance(observation, AIMessage):
            return observation.content, None
        if isinstance(observation, dict):
            result = observation.get("result")
            if isinstance(result, AIMessage):
                result = result.content
            return result, observation.get("references")
        return observation, None

    for action, observation in result['intermediate_steps']:
        tool_name = action.tool
        tool_query = _extract_tool_query(action)

        if tool_name == 'wikipedia_search':
            answer_text, references = _normalize_tool_output(observation)
            if answer_text:
                wikipedia_results.append({"query": tool_query, "answer": answer_text})
            if references:
                if isinstance(references, list):
                    state.references.extend(references)
                else:
                    state.references.append(references)
        elif tool_name == 'RAG_search':
            answer_text, references = _normalize_tool_output(observation)
            if answer_text:
                rag_results.append({"query": tool_query, "answer": answer_text})
            if references:
                if isinstance(references, list):
                    state.references.extend(references)
                else:
                    state.references.append(references)
        elif tool_name == 'ECOCROP_search':
            answer_text, _ = _normalize_tool_output(observation)
            if answer_text:
                if state.ecocrop_search_response:
                    state.ecocrop_search_response += "\n" + answer_text
                else:
                    state.ecocrop_search_response = answer_text
            if any("FAO, IIASA" not in element for element in state.references):
                state.references.append("FAO, IIASA: Global Agro-Ecological Zones (GAEZ V4) - Data Portal User's Guide, 1st edn. FAO and IIASA, Rome, Italy (2021). https://doi.org/10.4060/cb5167en")

    state.wikipedia_tool_response = wikipedia_results
    state.rag_search_response = rag_results
        
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
