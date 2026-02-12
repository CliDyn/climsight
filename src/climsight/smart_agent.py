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
    prompt = (
        "You are ClimSight's information gathering agent.\n"
        "Your task: collect background context about the user's query from external knowledge sources,\n"
        "then compile a structured summary for the downstream data analysis agents.\n\n"
        "## TOOLS\n\n"
        "1. **wikipedia_search** - Search Wikipedia for general context.\n"
        "   - Search strategy: (a) the location or region name, (b) the topic (crop, industry, hazard),\n"
        "     (c) the combination (e.g., 'agriculture in southern Germany').\n"
        "   - Call limit: max 10 calls total. If you reach 10, stop and use what you have.\n\n"
        "2. **RAG_search** - Query ClimSight's internal climate knowledge base.\n"
        "   - Use for: IPCC findings, peer-reviewed climate science, region-specific climate projections.\n"
        "   - Phrase queries as specific questions (e.g., 'What are projected temperature changes for Central Europe?').\n\n"
        "3. **ECOCROP_search** - Get crop-specific environmental requirements.\n"
        "   - ONLY use when the query explicitly mentions agriculture, crops, or food production.\n"
        "   - Do NOT use for general climate, infrastructure, or energy queries.\n\n"
    )
    if config['llm_smart']['model_type'] in ("local", "aitta"):
        prompt += (
            "**Tool use order:** Call tools one at a time, sequentially.\n\n"
        )
    else:
        prompt += (
            "**Tool use order:** Call wikipedia_search, RAG_search, and ECOCROP_search (if applicable)\n"
            "SIMULTANEOUSLY to gather information efficiently.\n\n"
        )
    prompt += (
        "## RULES\n\n"
        "- Focus on CONTEXT, not data extraction. Climate data retrieval is handled by other agents.\n"
        "- Include source references when citing Wikipedia or RAG results.\n"
        "- Present ECOCROP results verbatim (no citations needed for database facts).\n"
        "- Do NOT attempt to retrieve or analyze specific climate data values.\n"
        "- Do NOT include chain-of-thought reasoning or tool invocation details in your output.\n\n"
        "## OUTPUT FORMAT\n\n"
        "Provide a clean, organized summary with these sections (omit empty ones):\n\n"
        "1. **General Background** - key facts, context, qualitative+quantitative info with units\n"
        "2. **Scientific Context** - climate science findings, research results, projections\n"
        "3. **Specific Requirements** - ECOCROP thresholds and ranges (if applicable)\n"
        "4. **Key Takeaways** - 3-5 bullets summarizing the most important factors for analysis\n\n"
        "Present ONLY the final summary. No process explanation, no tool call narration.\n"
    )

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
        template = (
            "Read the provided Wikipedia article: {wikipage}\n\n"
            "Extract information relevant to the question: {question}\n\n"
            "Focus on these categories (adapt to the query topic — skip irrelevant ones, add domain-specific ones):\n"
            "Temperature, Precipitation, Wind, Elevation, Population, Natural Hazards, Soil Type,\n"
            "Energy potential, Infrastructure risks, Agricultural suitability, Water resources,\n"
            "Land use, Economic activity, Environmental regulations\n\n"
            "For each relevant category, separate into:\n"
            "- **Quantitative**: specific numbers, measurements, thresholds with units\n"
            "- **Qualitative**: descriptive context without specific numbers\n\n"
            "Example output format:\n\n"
            "• Temperature:\n"
            "  • Quantitative: Requires warm days above 10°C (50°F) for flowering. Frost-free period of 120-150 days.\n"
            "  • Qualitative: Cold-intolerant crop; must be planted after last spring frost in temperate zones.\n"
            "• Wind:\n"
            "  • Quantitative: Can be uprooted by winds exceeding 60 km/h due to shallow root system.\n"
            "  • Qualitative: Pollen is primarily dispersed by wind; strong winds during pollination reduce yield.\n"
            "• Soil Type:\n"
            "  • Quantitative: Optimal soil pH between 6.0 and 7.5. Requires minimum 200mm soil moisture.\n"
            "  • Qualitative: Intolerant of nutrient-deficient soils; performs best in deep, well-drained loams.\n\n"
            "Rules:\n"
            "- Omit categories with no relevant information in the article — do not mention them at all.\n"
            "- Include units for ALL numerical values (°C, mm, km/h, m, %, etc.).\n"
            "- Extract both location-specific facts AND general domain knowledge relevant to the query.\n"
            "- If the article discusses multiple locations, focus on information most relevant to the user's query.\n"
            "- Do not add information beyond what the article contains.\n"
            "- Do not include source citations for Wikipedia content.\n"
        )
        
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
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(loader.load)
                raw_documents = future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            return "Wikipedia search timed out after 120 seconds. Skipping this source."
        except Exception as e:
            return f"Wikipedia search failed: {str(e)}"

        if not raw_documents:
            return "No Wikipedia article found for the query."
        content_str = 'Wikipedia article: ' + raw_documents[0].page_content

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
        template = (
            "Read the provided climate knowledge base content: {rag_content}\n\n"
            "Extract information relevant to the question: {question}\n\n"
            "Focus on these categories (adapt to the query topic — skip irrelevant ones, add domain-specific ones):\n"
            "Temperature, Precipitation, Wind, Elevation, Population, Natural Hazards, Soil Type,\n"
            "Energy potential, Infrastructure risks, Agricultural suitability, Water resources,\n"
            "Land use, Economic activity, Environmental regulations\n\n"
            "For each relevant category, separate into:\n"
            "- **Quantitative**: specific numbers, measurements, thresholds with units\n"
            "- **Qualitative**: descriptive context without specific numbers\n\n"
            "Example output format:\n\n"
            "• Temperature:\n"
            "  • Quantitative: Projected warming of 1.5-2.0°C by 2050 under SSP2-4.5. Heat wave days increase by 15-25 days/year.\n"
            "  • Qualitative: Mediterranean-type warming pattern with strongest signal in summer months.\n"
            "• Precipitation:\n"
            "  • Quantitative: Winter precipitation increases by 5-15%. Summer precipitation decreases by 10-20%.\n"
            "  • Qualitative: Shift toward more intense but less frequent rainfall events.\n"
            "• Natural Hazards:\n"
            "  • Quantitative: 100-year flood return period reduced to 50-year under RCP8.5.\n"
            "  • Qualitative: Compound risks from drought-heat wave combinations becoming more likely.\n\n"
            "Rules:\n"
            "- Omit categories with no relevant information — do not mention them at all.\n"
            "- Include units for ALL numerical values (°C, mm, %, days/year, etc.).\n"
            "- If the content references IPCC reports or peer-reviewed studies, include the citation inline\n"
            "  (e.g., 'IPCC AR6, WG2, Section 12.4').\n"
            "- Extract both location-specific projections AND general climate science findings relevant to the query.\n"
            "- Distinguish between observations (past data) and projections (future scenarios).\n"
            "- Do not add information beyond what the source material contains.\n"
            "- If no relevant information found, return ONLY: 'No relevant information found for the query.'\n"
        )
        
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
    #python_repl_tool = create_python_repl_tool()

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
    tools = [rag_tool, ecocrop_tool, wikipedia_tool]

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
