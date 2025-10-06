# src/climsight/agents/researcher_agent.py

import os
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from langchain.agents import AgentExecutor

# --- Helper Function for AITTA Models ---
def get_aitta_chat_model(model_name, **kwargs):
    try:
        from aitta_client import Model, Client
        from aitta_client.authentication import APIKeyAccessTokenSource
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
    except Exception:
        return None

# --- Researcher Node Function ---
def researcher(state: dict, config: dict, api_key: str, api_key_local: str) -> dict:
    """Researcher node that handles Wikipedia, RAG, and ECOCROP searches."""
    
    # Initialize LLM
    temperature = 1 if "o1" in config['model_name_tools'] else 0

    if config['model_type'] == "local":
        llm = ChatOpenAI(openai_api_base="http://localhost:8000/v1", model_name=config['model_name_tools'], openai_api_key=api_key_local, temperature=temperature)
    elif config['model_type'] == "openai":
        llm = ChatOpenAI(openai_api_key=api_key, model_name=config['model_name_tools'], temperature=temperature)
    elif config['model_type'] == "aitta":
        llm = get_aitta_chat_model(config['model_name_tools'], temperature=temperature)

    # --- Tool Functions (Self-contained logic for this agent) ---

    def process_wikipedia_article(query: str) -> dict:
        """
        Searches Wikipedia for a given query and extracts information related to climate, environment, and geography.
        It returns both the extracted information and a reference to the Wikipedia article.
        """
        template = """Read the provided {wikipage} carefully. Extract and present information related to the following keywords relative to {question}:
            • Temperature • Precipitation • Wind • Elevation Above Sea Level • Population • Natural Hazards • Soil Type
        Guidelines:
            • For each keyword, if information is available in the article, extract and present both qualitative and quantitative information separately.
            • If no information is available for a keyword, omit it entirely.
        Example Format:
            • Temperature:
                • Quantitative: Requires warm days above 10°C (50°F) for flowering.
                • Qualitative: Maize is cold-intolerant and must be planted in the spring in temperate zones."""
        prompt = ChatPromptTemplate.from_template(template)
        loader = WikipediaLoader(query=query, load_all_available_meta=True, doc_content_chars_max=100000, load_max_docs=1)
        raw_documents = loader.load()
        if not raw_documents: return {"result": "No Wikipedia article found for the query.", "references": []}
        content_str = 'Wikipedia article: ' + raw_documents[0].page_content
        chain = prompt | llm
        result = chain.invoke({"wikipage": content_str, "question": query})
        title = raw_documents[0].metadata.get("title", "").replace(" ", "_")
        ref = "Wikipedia URL: " + f"https://en.wikipedia.org/wiki/{title}"
        return {"result": result.content, "references": [ref]}

    def process_RAG_search(query: str) -> dict:
        """
        Searches an internal knowledge base (RAG) for specific climate-related information.
        This tool is useful for finding detailed, domain-specific data not available on Wikipedia.
        """
        data_rag = config['rag_articles']['data_path']
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = Chroma(persist_directory=data_rag, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs: return {"result": "No relevant documents found for the query.", "references": []}
        rag_content = '\n\n'.join([doc.page_content for doc in retrieved_docs])
        rag_references = [doc.metadata.get("source", "") for doc in retrieved_docs]
        template = """Read the provided {rag_content} carefully. Extract and present information related to the following keywords relative to {question}:
            • Temperature • Precipitation • Wind • Elevation Above Sea Level • Population • Natural Hazards • Soil Type
        Guidelines:
            • Extract both qualitative and quantitative information.
            • If no information is found, just return 'No relevant information found for the query.' AND NOTHING ELSE."""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        result = chain.invoke({"rag_content": rag_content, "question": query})
        return {"result": result.content, "references": rag_references}

    def process_ecocrop_search(query: str) -> str:
        """
        Searches the ECOCROP database for environmental and soil requirements of a specific crop.
        The input should be the name of the crop (e.g., "corn", "wheat").
        """
        ecocroploc = config['ecocrop']['ecocroploc_path']
        variable_expl_path = config['ecocrop']['variable_expl_path']
        rag_ecocrop = config['ecocrop']['data_path']
        ecocropall = pd.read_csv(ecocroploc, encoding='latin1', engine="python")
        variable_expl = pd.read_csv(variable_expl_path, engine='python')
        variable_dict = dict(zip(variable_expl['Variable'], variable_expl['Explanation']))
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = Chroma(embedding_function=embeddings, persist_directory=rag_ecocrop)
        prompt = ChatPromptTemplate.from_template("Select the most relevant document for the query: {query}\nDocuments:\n{documents}\nProvide ONLY the SOURCE of the single most relevant one.")
        query_result = vector_store.similarity_search(query=query, k=10)
        if not query_result: return f"No data found for {query}."
        messages = prompt.format_messages(query=query, documents=query_result)
        response = llm.invoke(messages)
        filtered_data = ecocropall[ecocropall['ScientificName'] == response.content.strip()]
        if filtered_data.empty: return f"No data found for {query}."
        output = f"Data from ECOCROP for {query}:\n"
        output += f"Scientific Name: {filtered_data['ScientificName'].iloc[0]}\n"
        for varname, explanation in variable_dict.items():
            value = filtered_data[varname].iloc[0]
            if pd.notna(value):
                output += f"{explanation}: {value}\n"
        return output

    # --- Tool Creation ---
    class WikipediaSearchArgs(BaseModel): query: str = Field(description="The topic to search on Wikipedia.")
    wikipedia_tool = StructuredTool.from_function(func=process_wikipedia_article, name="wikipedia_search", args_schema=WikipediaSearchArgs)

    class RAGSearchArgs(BaseModel): query: str = Field(description="The topic to search in the internal knowledge database.")
    rag_tool = StructuredTool.from_function(func=process_RAG_search, name="RAG_search", args_schema=RAGSearchArgs)

    class EcoCropSearchArgs(BaseModel): query: str = Field(description="The name of the crop to search for in the ECOCROP database.")
    ecocrop_tool = StructuredTool.from_function(func=process_ecocrop_search, name="ECOCROP_search", args_schema=EcoCropSearchArgs)
    
    #research_tools = [wikipedia_tool, rag_tool, ecocrop_tool]
    research_tools = [rag_tool, ecocrop_tool]
    
    # --- Agent Definition ---
    from langchain.prompts import PromptTemplate

    # This is a standard ReAct prompt template that includes all required variables.
    # We are defining it directly to ensure it has the correct structure.
    REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    agent_prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    # Create agent and executor
    agent = create_react_agent(llm, research_tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=research_tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True  # Helps prevent crashes if the LLM formats its output incorrectly
    )
    
    # Get the latest user message
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    # Run the agent
    result = agent_executor.invoke({"input": user_query})
    
    # Extract tool outputs and compile response
    tool_outputs = []
    references = []
    
    for action, observation in result['intermediate_steps']:
        if action.tool == 'wikipedia_search' and isinstance(observation, dict):
            tool_outputs.append(f"Wikipedia Search:\n{observation.get('result', '')}")
            refs = observation.get('references', [])
            if isinstance(refs, list):
                references.extend(refs)
            elif isinstance(refs, str):
                references.append(refs)
        elif action.tool == 'RAG_search' and isinstance(observation, dict):
            tool_outputs.append(f"RAG Search:\n{observation.get('result', '')}")
            refs = observation.get('references', [])
            if isinstance(refs, list):
                references.extend(refs)
            elif isinstance(refs, str):
                references.append(refs)
        elif action.tool == 'ECOCROP_search':
            tool_outputs.append(f"ECOCROP Search:\n{observation}")
            if "FAO, IIASA" not in str(references):
                references.append("FAO, IIASA: Global Agro-Ecological Zones (GAEZ V4) - Data Portal User's Guide, 1st edn. FAO and IIASA, Rome, Italy (2021). https://doi.org/10.4060/cb5167en")
    
    # Compile response
    response_content = result['output']
    if tool_outputs:
        response_content += "\n\nDetailed findings:\n" + "\n\n".join(tool_outputs)
    if references:
        response_content += "\n\nReferences:\n" + "\n".join(set(references))
    
    # Return updated state
    new_message = {"role": "assistant", "content": response_content, "name": "researcher"}
    return {
        "messages": state["messages"] + [new_message]
    }