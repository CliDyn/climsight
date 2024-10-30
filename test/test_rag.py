import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock

# Append the directory containing the module to sys.path
module_dir = os.path.abspath('../src/climsight/')
if module_dir not in sys.path:
    sys.path.append(module_dir)

# make sure the config file gets found (error if line is missing)
os.environ['CONFIG_PATH'] = os.path.abspath('../config.yml')

# Import module functions
from rag import (
    load_rag
)

from langchain_core.documents import Document


class TestLoadRag(unittest.TestCase):
    @patch('rag.is_valid_rag_db', return_value=True) # simulate case where db is valid
    @patch('rag.Chroma')
    @patch('rag.OpenAIEmbeddings')
    def test_load_rag_when_ready(self, mock_openai_embeddings, mock_chroma, mock_is_valid_rag_db):
        # Mock the embeddings and chroma instances
        mock_embedding_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embedding_instance
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        # Call the function with the mocks
        rag_ready, rag_db = load_rag(embedding_model='text-embedding-3-large', chroma_path='test_chroma_path', openai_api_key='test_key')

        # Assert that OpenAIEmbeddings and Chroma were called when rag_ready is True
        self.assertTrue(rag_ready)
        self.assertIs(rag_db, mock_chroma_instance) # making sure that rag_db is exactly the mocked chroma instance
        mock_openai_embeddings.assert_called_once_with(openai_api_key='test_key', model='text-embedding-3-large')
        mock_chroma.assert_called_once_with(
            persist_directory='test_chroma_path',
            embedding_function=mock_embedding_instance,
            collection_name="ipcc_collection"
        )

    @patch('rag.is_valid_rag_db', return_value=False) # simulate case where db is not valid
    def test_load_rag_when_not_ready(self, mock_is_valid_rag_db):
        # Call the function without valid RAG setup
        rag_ready, rag_db = load_rag(embedding_model='text-embedding-3-large', chroma_path='test_chroma_path', openai_api_key='test_key')

        # Assertions for an invalid RAG database
        self.assertFalse(rag_ready)
        self.assertIsNone(rag_db)
        

# Run the tests
if __name__ == '__main__':
    unittest.main()
