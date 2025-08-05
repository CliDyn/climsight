# Database generation based on source (text) files
# only gets executed if run actively and separately

import os
import logging
import tqdm
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

# Import the new embedding utility
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from climsight.embedding_utils import create_embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='db_generation.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Load config
config_path = os.getenv('CONFIG_PATH', 'config.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


def uuid_patternn():
    """Returns a regex pattern for matching any UUID folder name."""
    return re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")


def is_valid_rag_db(rag_db_path):
    """Checks if the rag_db folder contains chroma.sqlite3 and non-empty UUID folder."""
    # check for chroma.sqlite3
    chroma_file = os.path.join(rag_db_path, 'chroma.sqlite3')
    if not os.path.exists(chroma_file):
        return False
    # check for non-empty folder with UUID name
    uuid_folder = [f for f in os.listdir(rag_db_path) if os.path.isdir(os.path.join(rag_db_path, f)) and uuid_patternn().match(f)]
    for file in uuid_folder:
        folder_path = os.path.join(rag_db_path, file)
        if os.listdir(folder_path): # check if folder is non-empty
            return True
        
    return False


def are_source_files_available(data_path):
    """Checks if the ipcc_text_reports folder exists and is non-empty."""
    ipcc_reports_path = os.path.join(data_path, 'ipcc_text_reports')
    if not os.path.exists(ipcc_reports_path):
        return False
    # check for non-hidden files
    for file in os.listdir(ipcc_reports_path):
        if not file.startswith('.'):
            return True
        
    # if only hidden files or folder is empty
    return False


def get_file_names(folder_path):
    """
    Gets the names of all text files in a folder. Throws an error if the
    folder contains other files beyond .txt.

    Params:
    folder_path (str): name of the folder where all files are stored.

    Returns:
    file_names (list): list of all file names.
    """
    file_names = []
    for filename in os.listdir(folder_path):
        # ignore hidden files and folders (those that start with '.')
        if filename.startswith('.'):
            logger.info(f"Ignoring hidden file or folder: {filename}")
            continue
        # only allow .txt files
        if filename.endswith('.txt'):
            file_names.append(filename)
        else:
            # Log and raise an error if a non-text file is found
            logger.error(f"Non-text file found: {filename}")
            raise ValueError(f"Non-text file found: {filename}")
    return file_names


def load_docs(file, encoding='utf-8'):
    """
    Loads (and decodes) a text file with langchain textLoader.

    Params:
    file (str): name of the file that is being loaded.
    encoding (str): type of encoding of the text file. Default: utf-8

    Returns:
    documents (list): list of documents loaded
    """
    if not file.endswith('.txt'):
        logger.error(f"Failed to load {file}: Not a text file.")
        return []  # Return an empty list for non-text files

    try:
        loader = TextLoader(file, encoding=encoding, autodetect_encoding=True)  # autodetect encoding is essential!
        documents = loader.load()
        return documents

    except Exception as e:
        logger.error(f"Failed to load {file}: {e}")
        return []


def split_docs(documents, chunk_size=2000, chunk_overlap=200, separators=[" ", ",", "\n"]):
    """
    Splits the passed documents into chunks.

    Params:
    documents (list): list of document objects to be split.
    chunk_size (int): maximum number of characters per chunk. Default: 2000.
    chunk_overlap (int): number of characters to overlap per chunk. Default: 200.
    separators (list): list of characters where text can be split. Default: [" ", ",", "\n"]

    Returns:
    docs (list): list of chunks created from input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    docs = text_splitter.split_documents(documents)
    return docs


def chunk_documents(document_path, chunk_size=2000, chunk_overlap=200, separators=[" ", ",", "\n"]):
    """
    Chunks and documents from the specified directory.

    Args:
    - document_path (str): The path to the directory containing the documents.
    - chunk_size (int): maximum number of characters per chunk. Default: 2000.
    - chunk_overlap (int): number of characters to overlap per chunk. Default: 200.
    - separators (list): list of characters where text can be split. Default: [" ", ",", "\n"]

    Returns:
    - list: A list of chunked documents.
    """
    # load documents
    file_names = get_file_names(document_path)
    all_documents = []
    for file in file_names:
        logger.info(f"Processing file: {file}")
        documents = load_docs(os.path.join(document_path, file))
        all_documents.extend(documents)  # save all of them into one

    if not all_documents:
        logger.info("No documents found for chunking.")
        return []

    # Chunk documents
    chunked_docs = split_docs(
        documents=all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    logger.info(f"Chunked documents into {len(chunked_docs)} pieces.")
    return chunked_docs


def initialize_rag(config):
    """
    Initializes the RAG database by checking document presence and modification times,
    and performs chunking and embedding if necessary.

    Args:
    - config (dict): configuration dictionary

    Returns:
    - rag_ready (bool): true if RAG database is initialized successfully, false otherwise
    """
    rag_ready = False

    rag_settings = config['rag_settings']
    embedding_model_type = rag_settings.get('embedding_model_type', 'openai')
    # Select embedding model and chroma path based on type
    if embedding_model_type == 'openai':
        embedding_model = rag_settings.get('embedding_model_openai')
        chroma_path = rag_settings.get('chroma_path_ipcc_openai')
    elif embedding_model_type == 'aitta':
        embedding_model = rag_settings.get('embedding_model_aitta')
        chroma_path = rag_settings.get('chroma_path_ipcc_aitta')
    # Add more types here as needed
    # elif embedding_model_type == 'mistral':
    #     embedding_model = rag_settings.get('embedding_model_mistral')
    #     chroma_path = rag_settings.get('chroma_path_ipcc_mistral')
    else:
        raise ValueError(f"Unknown embedding_model_type: {embedding_model_type}")

    openai_api_key = os.getenv('OPENAI_API_KEY')
    aitta_api_key = os.getenv('AITTA_API_KEY')
    aitta_url = rag_settings.get('aitta_url', os.getenv('AITTA_URL', 'https://api-climatedt-aitta.2.rahtiapp.fi'))
    document_path = rag_settings['document_path']
    chunk_size = rag_settings['chunk_size']
    chunk_overlap = rag_settings['chunk_overlap']
    separators = rag_settings['separators']

    # check if api key there (either OpenAI or AITTA)
    if embedding_model_type == 'openai' and not openai_api_key:
        logger.warning("No OpenAI API Key found. Skipping RAG initialization.")
        return False
    if embedding_model_type == 'aitta' and not aitta_api_key:
        logger.warning("No AITTA API Key found. Skipping RAG initialization.")
        return False

    # check if documents are present and valid
    if not os.path.exists(document_path) or not any(file.endswith('.txt') for file in os.listdir(document_path)):
        logger.warning("No valid documents found in the specified path. Skipping RAG initialization.")
        return False

    # Perform chunking and embedding
    try:
        langchain_ef = create_embeddings(
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            aitta_api_key=aitta_api_key,
            aitta_url=aitta_url,
            model_type=embedding_model_type
        )
        documents = chunk_documents(document_path, chunk_size, chunk_overlap, separators)
        rag_db = Chroma(
            collection_name="ipcc_collection",
            persist_directory=chroma_path,
            embedding_function=langchain_ef
        )
        batch_size = 32
        for i in tqdm.tqdm(range(0, len(documents), batch_size)):
            rag_db.add_documents(documents[i:i+batch_size])
        rag_ready = True
        logger.info(f"RAG ready: {rag_ready}")
        logger.info("RAG database has been initialized and documents embedded.")
        return rag_ready
    except Exception as e:
        logger.error(f"Failed to initialize the RAG database: {e}")
        return False


def main():
    # paths
    rag_db_path = './rag_db' # using static paths as they are not supposed to be changed and will remain the same.
    data_path = './data'
    rag_ready = False

    if is_valid_rag_db(rag_db_path):
        logger.info("RAG database already exists. No need to initialize.")
        rag_ready = True
        return rag_ready
    
    if not are_source_files_available(data_path):
        logger.warning("""The RAG database does not exists yet and there are no source files available in the data/ipcc_text_reports folder.
                       Please run the download_data.py again and make sure to set the flag --source_files=True.""")
        return rag_ready

    # if rag does not exist yet and the source files are available, run the initialization
    logger.info("Initializing RAG database...")
    rag_ready = initialize_rag(config)
    # danach kann es den Fall geben, dass es trotzdem keine db gibt, weil bei der Initialisierung etwas falsch gelaufen ist (zb. pdf statt txt files in ipcc_text_reports folder)
    # TO-DO write a test for this case!

if __name__ == "__main__":
    main()
