# Database generation based on source (text) files
# only gets executed if run actively and separately

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


def chunk_and_embed_documents(document_path, embedding_model, openai_api_key, chunk_size=2000, chunk_overlap=200, separators=[" ", ",", "\n"]):
    """
    Chunks and embeds documents from the specified directory using provided embedding function.

    Args:
    - document_path (str): The path to the directory containing the documents.
    - embedding_model (OpenAIEmbeddings): The embedding function to use for generating embeddings.
    - chunk_size (int): maximum number of characters per chunk. Default: 2000.
    - chunk_overlap (int): number of characters to overlap per chunk. Default: 200.
    - separators (list): list of characters where text can be split. Default: [" ", ",", "\n"]

    Returns:
    - list: A list of documents with embeddings.
    """
    # load documents
    file_names = get_file_names(document_path)
    all_documents = []
    for file in file_names:
        logger.info(f"Processing file: {file}")
        documents = load_docs(os.path.join(document_path, file))
        all_documents.extend(documents)  # save all of them into one

    if not all_documents:
        logger.info("No documents found for chunking and embedding.")
        return []

    # Chunk documents
    chunked_docs = split_docs(
        documents=all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    logger.info(f"Chunked documents into {len(chunked_docs)} pieces.")

    # embedding documents 
    embedded_docs = []
    embedding_item = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model) 
    try:
        for doc in chunked_docs:
            embedding = embedding_item.embed_documents([doc.page_content])[0]  # embed_documents returns a list, so we take the first element
            embedded_docs.append({"text": doc.page_content, "embedding": embedding, "metadata": doc.metadata})
    except Exception as e:
        logger.error(f"Failed to embed document chunks: {e}")
        return []

    logger.info(f"Embedded {len(embedded_docs)} document chunks.")
    return embedded_docs


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
    openai_api_key = os.getenv('OPENAI_API_KEY')
    embedding_model = rag_settings['embedding_model']
    chroma_path = rag_settings['chroma_path']
    document_path = rag_settings['document_path']
    chunk_size = rag_settings['chunk_size']
    chunk_overlap = rag_settings['chunk_overlap']
    separators = rag_settings['separators']

    # check if api key there
    if not openai_api_key:
        logger.warning("No OpenAI API Key found. Skipping RAG initialization.")
        rag_ready = False
        return rag_ready

    # check if documents are present and valid
    if not os.path.exists(document_path) or not any(file.endswith('.txt') for file in os.listdir(document_path)):
        logger.warning("No valid documents found in the specified path. Skipping RAG initialization.")
        rag_ready = False
        return rag_ready

    # Perform chunking and embedding
    try:
        # embedding function, using the langchain chroma package (not chromadb directly)
        langchain_ef = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model)  # max_retries, request_timeout, retry_min_seconds
        documents = chunk_and_embed_documents(document_path, embedding_model, openai_api_key, chunk_size, chunk_overlap, separators)
        converted_documents = [
            Document(page_content=doc['text'], metadata=doc['metadata'])
            for doc in documents
        ]
        rag_db = Chroma.from_documents(
            documents=converted_documents,
            persist_directory=chroma_path,
            embedding=langchain_ef,
            collection_name="ipcc_collection"
        )
        rag_ready = True
        logger.info(f"RAG ready: {rag_ready}")
        logger.info("RAG database has been initialized and documents embedded.")

        return rag_ready

    except Exception as e:
        logger.error(f"Failed to initialize the RAG database: {e}")
        return rag_ready


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
