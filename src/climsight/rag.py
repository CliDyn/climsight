import os
import time
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# global variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
CHROMA_PATH = "rag_db"

# local variables
document_path = './data/ipcc_text_reports/'
timestamp_file = './data/last_update.txt'

def get_file_mod_times(folder_path):
    """
    Gets the modification times of all files in the folder.

    Params:
    folder_path (str): Path to the folder.

    Returns:
    mod_times (dict): Dictionary of filenames and their modification times.
    """
    mod_times = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            full_path = os.path.join(folder_path, file)
            mod_times[file] = os.path.getmtime(full_path)
    return mod_times


def initialize_rag_database(force_update=False):
    # check if api key is available in environment; 
    # otherwise no rag initialization (might still use rag if already exists)
    if not OPENAI_API_KEY:
        print("No OpenAI API Key found. Skipping RAG database initialization.")
        return
    
    all_documents = []


    def get_file_names(folder_path):
        """
        Gets the names of all text files in a folder. Throws an error if the
        folder contains other files beyond .txt.

        Params:
        folder_path (str): name of the folder where all files are stored.

        Returns:
        file_names (list): list of all file names.
        """
        try:
            # Check if all files are text files
            for file in os.listdir(folder_path):
                if not file.endswith('.txt'):
                    raise ValueError(f"Non-text file found: {file}")

            # get names of all files
            files = os.listdir(folder_path)
            file_names = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
            return file_names

        except Exception as e:
            print(f"Error: {e}")
            return []

    # check if the database needs to be updated
    current_mod_times = get_file_mod_times(document_path)
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            last_mod_times = eval(f.read())
    else:
        last_mod_times = {}

    if not force_update and current_mod_times == last_mod_times:
        print("No changes detected in documents. Skipping database initialization.")
        return

    file_names = get_file_names(document_path)
    #print(file_names)


    def load_docs(file, encoding='utf-8'):
        """
        Loads (and decodes) a text file with langchain textLoader.

        Params:
        file (str): name of the file that is being loaded.
        encoding (str): type of encoding of the text file. Default: utf-8

        Returns:
        documents (list): list of documents loaded
        """
        loader = TextLoader(file, encoding=encoding, autodetect_encoding=True)  # autodetect encoding is essential!
        documents = loader.load()
        return documents


    for file in file_names:
        print(file)
        try:
            documents = load_docs(os.path.join(document_path, file))
            all_documents.extend(documents)  # save all of them into one
        except RuntimeError as e:
            print(f"Failed to load {file}: {e}")


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


    docs = split_docs(documents=all_documents)

    try:
        # embedding function, using the langchain chroma package (not chromadb directly)
        langchain_ef = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)  # max_retries, request_timeout, retry_min_seconds

        # save it into Chroma (disk)
        db = Chroma.from_documents(persist_directory=CHROMA_PATH, collection_name="ipcc-collection", documents=docs, embedding=langchain_ef)
        logger.info(f"There are {db._collection.count()} entries in the collection")

        # Update the timestamp file
        with open(timestamp_file, 'w') as f:
            f.write(str(current_mod_times))
    except Exception as e:
        logger.error(f"Failed to initialize the RAG database: {e}")



# call function to initialize database
if __name__ == "__main__":
    initialize_rag_database()
