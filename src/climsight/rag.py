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

logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Load config
config_path = os.getenv('CONFIG_PATH', 'config.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def get_folder_name(rag_db_path):
    """
    Extracts and returns part of the folder type (ipcc or general) from the rag_db_path that is needed for later distingushing.
    """
    if "ipcc_reports" in rag_db_path:
        return "ipcc"
    elif "general_reports" in rag_db_path:
        return "general"
    else:
        return None


def is_valid_rag_db(rag_db_path):
    """Checks if the rag_db folder contains chroma.sqlite3 and non-empty UUID folder."""
    # check for chroma.sqlite3
    chroma_file = os.path.join(rag_db_path, 'chroma.sqlite3')
    if not os.path.exists(chroma_file):
        return False
    folder_name = get_folder_name(rag_db_path)
    if folder_name is None:
        return False
    folder_path = os.path.join(rag_db_path, folder_name)
    if os.path.isdir(folder_path) and os.listdir(folder_path): # check if folder is non-empty
        return True
        
    return False


def load_rag(embedding_model, chroma_path, openai_api_key):
    """
    Loads the RAG database if it has been initialized before and is ready to use.

    Args:
    - embedding_model (str): Name of the embedding model to be used for loading embeddings.
    - chroma_path (str): Path where the Chroma database is stored.
    - openai_api_key (str): OpenAI API Key

    Returns:
    - tuple (bool, Chroma or None): 
        - rag_ready (bool): true if the RAG database was successfully loaded, false otherwise.
        - rag_db (Chroma or None): The loaded Chroma database object if successful, None if loading failed.

    """
    rag_ready = False
    valid_rag_db = is_valid_rag_db(chroma_path)

    if not valid_rag_db:
        logger.warning("RAG database is not valid. Not loading it. Please run 'python db_generation.py first.")
        rag_db = None
        return rag_ready, rag_db

    try:
        langchain_ef = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model)
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
            logger.info(state)
            return state
        rag_chain = (
            {"context": retriever | format_docs, "location": RunnableLambda(get_loci), "question": RunnablePassthrough()}
            | RunnableLambda(inspect)
            | custom_rag_prompt
            | ChatOpenAI(model=config['model_name'], api_key=openai_api_key)
            | StrOutputParser()
        )
        rag_response = rag_chain.invoke(input_params['user_message'])
        logging.info(f"RAG response: {rag_response}")

        return rag_response

    except Exception as e:
        logger.error(f"Failed to perform RAG query: {e}")
        return None
