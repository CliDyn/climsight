import os
import logging
import yaml
import re

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document

from embedding_utils import create_embeddings
from smart_agent import get_aitta_chat_model

logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


def is_valid_rag_db(rag_db_path):
    """Checks if the rag_db folder contains chroma.sqlite3 and non-empty UUID folder."""

    # check for chroma.sqlite3
    chroma_file = os.path.join(rag_db_path, 'chroma.sqlite3')
    if not os.path.exists(chroma_file):
        return False
    # check for a nonempty folder
    for filename in os.listdir(rag_db_path):
        filepath = os.path.join(rag_db_path, filename)
        if os.path.isdir(filepath) and os.listdir(filepath):
            return True
    return False


def load_rag(config, openai_api_key=None, db_type='ipcc'):
    """
    Loads the RAG database if it has been initialized before and is ready to use.

    Args:
    - config (dict): Configuration dictionary.
    - openai_api_key (str): OpenAI API Key (do not overwrite with os.getenv)
    - db_type (str): 'ipcc' or 'general' (determines which DB to load)

    Returns:
    - tuple (bool, Chroma or None): 
        - rag_ready (bool): true if the RAG database was successfully loaded, false otherwise.
        - rag_db (Chroma or None): The loaded Chroma database object if successful, None if loading failed.
    """
    rag_settings = config['rag_settings']
    embedding_model_type = rag_settings.get('embedding_model_type', 'openai')
    # Select embedding model and chroma path based on type and db_type
    if embedding_model_type == 'openai':
        embedding_model = rag_settings.get('embedding_model_openai')
        chroma_path = rag_settings.get(f'chroma_path_{db_type}_openai')
    elif embedding_model_type == 'aitta':
        embedding_model = rag_settings.get('embedding_model_aitta')
        chroma_path = rag_settings.get(f'chroma_path_{db_type}_aitta')
    # Add more types here as needed
    # elif embedding_model_type == 'mistral':
    #     embedding_model = rag_settings.get('embedding_model_mistral')
    #     chroma_path = rag_settings.get(f'chroma_path_{db_type}_mistral')
    else:
        raise ValueError(f"Unknown embedding_model_type: {embedding_model_type}")

    # Use the openai_api_key parameter as-is (do not overwrite)
    aitta_api_key = os.getenv('AITTA_API_KEY')
    aitta_url = rag_settings.get('aitta_url', os.getenv('AITTA_URL', 'https://api-climatedt-aitta.2.rahtiapp.fi'))

    rag_ready = False
    valid_rag_db = is_valid_rag_db(chroma_path)
    if not valid_rag_db:
        logger.warning("RAG database is not valid. Not loading it. Please run 'python db_generation.py' first.")
        rag_db = None
        return rag_ready, rag_db

    try:
        langchain_ef = create_embeddings(
            model_type=embedding_model_type,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            aitta_api_key=aitta_api_key,
            aitta_url=aitta_url
        )
        rag_db = Chroma(persist_directory=chroma_path, embedding_function=langchain_ef, collection_name="ipcc_collection")
        logger.info(f"RAG database loaded with {rag_db._collection.count()} documents.")
        rag_ready = True
    except Exception as e:
        logger.warning(f"Failed to load the RAG database: {e}")
        rag_db = None
        rag_ready = False

    return rag_ready, rag_db


def format_docs(docs):
    """
    Formats the retrieved documents into a single string.

    Params:
    docs (list): List of documents retrieved by the RAG database.

    Returns:
    str: Formatted string of the document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def extract_sources(docs):
    """
    Extracts the 'source' metadata from each document and returns them as a list.
    
    Args:
        docs (list): List of Document objects.
        
    Returns:
        list: A list containing the source of each document.
    """
    return [doc.metadata.get("source", "Unknown") for doc in docs]

def query_rag(input_params, config, openai_api_key, rag_ready, rag_db):
    """
    Queries the RAG database with the user's input.

    Args:
    - input_params (dict): The user's input parameters.
    - config (dict): Configuration dictionary.
    - openai_api_key (str): OpenAI API Key.
    - rag_ready (bool): Boolean flag indicating whether the RAG database is loaded and ready.
    - rag_db (Chroma or None): The loaded RAG database object.

    Returns:
    - str or None: The response from the RAG query, or None if the query fails.
    """
    if not rag_ready:
        logger.warning("RAG database is not ready or loaded. Skipping RAG query.")
        return None
    try:
        retriever = rag_db.as_retriever()
        if retriever is None:
            logger.error("Failed to create retriever: retriever is None.")

        # load template from config
        template = config['rag_template']
        custom_rag_prompt = PromptTemplate.from_template(template)

        location = input_params['location_str']

        def get_loci(_):
            return location

        # inspect chain - just for development
        def inspect(state):
            """Print the state passed between Runnables in a langchain and pass it on"""
            logger.info(f"Chunks returned from RAG: {state}")
            return state
        
        
        # # Retrieve documents based on the user's message
        #docs = list(retriever.get_documents(input_params['user_message']))
        #docs = list(retriever.get_relevant_documents(input_params['user_message']))
        # # Extract sources from the documents
        #sources_list = extract_sources(docs)
        # # Format the document content into a single string
        # context = format_docs(docs)
        # Build chain input as a dictionary.
        # # Note: In this chain, we only pass the context, location, and question to the prompt.
        # # The sources are kept separately.
        # chain_input = {
        #     "context": context,
        #     "location": get_loci(None),
        #     "question": input_params['user_message']
        # }
        
        # # Build the chain
        # rag_chain = (
        #     chain_input
        #     | RunnableLambda(inspect)
        #     | custom_rag_prompt
        #     | ChatOpenAI(model=config['model_name'], api_key=openai_api_key)
        #     | StrOutputParser()
        # )   
        llm = None
        if config['model_type'] == 'openai':
            llm = ChatOpenAI(model=config['model_name_rag'], api_key=openai_api_key)
        elif config['model_type'] == 'aitta':
            llm = get_aitta_chat_model(config['model_name_rag'])
        else:
            raise NotImplementedError(f"Unknown model type: {config['model_type']}")
        rag_chain = (
            {"context": retriever | format_docs, "location": RunnableLambda(get_loci), "question": RunnablePassthrough()}
            | RunnableLambda(inspect)
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        rag_response = rag_chain.invoke(input_params['user_message'])
        logging.info(f"Rendered RAG response: {rag_response}")

        return rag_response

    except Exception as e:
        logger.error(f"Failed to perform RAG query: {e}")
        return None
