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
    get_file_mod_times,
    get_file_names,
    load_docs,
    split_docs,
    chunk_and_embed_documents,
    initialize_rag,
    load_rag,
)

from langchain_core.documents import Document


class TestGetFileModTimes(unittest.TestCase):
    def test_get_file_mod_times(self):
        # Create a temporary directory with some text files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file1 = os.path.join(temp_dir, "file1.txt")
            temp_file2 = os.path.join(temp_dir, "file2.txt")

            # Write some data to the files
            with open(temp_file1, 'w') as f:
                f.write("Hello, World!")

            with open(temp_file2, 'w') as f:
                f.write("Hello, again!")

            # Get modification times
            mod_times = get_file_mod_times(temp_dir)
            print(f"from function: {mod_times}")

            # Assert that modification times are returned for each file
            self.assertIn("file1.txt", mod_times)
            self.assertIn("file2.txt", mod_times)
            self.assertGreater(mod_times["file1.txt"], 0)
            self.assertGreater(mod_times["file2.txt"], 0)

    def test_get_file_mod_times_non_text_files(self):
        # Create a temporary directory with non-text files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "file1.pdf")

            # Write some data to the file
            with open(temp_file, 'w') as f:
                f.write("Hello, World!")

            # Get modification times
            mod_times = get_file_mod_times(temp_dir)

            # Assert that non-text files are not included
            self.assertNotIn("file1.pdf", mod_times)


class TestGetFileNames(unittest.TestCase):
    def test_get_file_names_with_text_files(self):
        # Create a temporary directory with some text files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file1 = os.path.join(temp_dir, "file1.txt")
            temp_file2 = os.path.join(temp_dir, "file2.txt")

            # Write some data to the files
            with open(temp_file1, 'w') as f:
                f.write("Hello, World!")

            with open(temp_file2, 'w') as f:
                f.write("Hello, again!")

            # Get file names
            file_names = get_file_names(temp_dir)

            # Assert that all text files are returned
            self.assertEqual(set(file_names), {"file1.txt", "file2.txt"})

    def test_get_file_names_with_non_text_files(self):
        # Create a temporary directory with mixed file types
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file1 = os.path.join(temp_dir, "file1.txt")
            temp_file2 = os.path.join(temp_dir, "file2.pdf")

            # Write some data to the files
            with open(temp_file1, 'w') as f:
                f.write("Hello, World!")

            with open(temp_file2, 'w') as f:
                f.write("Hello, again!")

            # Try to get file names
            try:
                file_names = get_file_names(temp_dir)
            except ValueError as e:
                self.assertEqual(str(e), "Non-text file found: file2.pdf")
            else:
                self.fail("ValueError not raised")

    def test_get_file_names_empty_folder(self):
        # Create a temporary empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get file names
            file_names = get_file_names(temp_dir)

            # Assert that an empty list is returned
            self.assertEqual(file_names, [])


class TestLoadDocs(unittest.TestCase):
    def test_load_docs_with_valid_file(self):
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(b"Hello, World!")

        # Load documents
        documents = load_docs(temp_file.name)

        # Assert that documents are loaded correctly
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Hello, World!")

    def test_load_docs_with_invalid_file(self):
        # Create a temporary non-text file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(b"%PDF-1.4")

        # Load documents
        documents = load_docs(temp_file.name)

        # Assert that no documents are loaded
        self.assertEqual(len(documents), 0)


class TestSplitDocs(unittest.TestCase):
    def test_split_docs(self):
        # Create a mock document
        documents = [Document(page_content="This is a test document. " * 100)]

        # Split documents
        chunks = split_docs(documents, chunk_size=50, chunk_overlap=10)

        # Assert that chunks are created correctly
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(chunk.page_content) <= 50 for chunk in chunks))


class TestChunkAndEmbedDocuments(unittest.TestCase):
    @patch('rag.OpenAIEmbeddings')
    def test_chunk_and_embed_documents(self, mock_embeddings):
        # Create a temporary directory for the document path
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary text file in this directory
            temp_file_path = os.path.join(temp_dir, "file1.txt")
            with open(temp_file_path, 'w') as f:
                f.write("This is some test content.")
            # mock the embeddings
            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding values
            mock_embeddings.return_value = mock_embedding_instance

            # Call the function
            embedded_docs = chunk_and_embed_documents(
                document_path=temp_dir,
                embedding_model=mock_embedding_instance,
                chunk_size=50,
                chunk_overlap=10
            )

            # Assert the function returns the expected results
            self.assertEqual(len(embedded_docs), 1)
            self.assertEqual(embedded_docs[0]['text'], "This is some test content.")
            self.assertEqual(embedded_docs[0]['embedding'], [0.1, 0.2, 0.3])


class TestInitializeRag(unittest.TestCase):
    @patch('rag.get_file_mod_times')
    @patch('rag.chunk_and_embed_documents')
    @patch('rag.Chroma')
    def test_initialize_rag_no_changes(self, mock_chroma, mock_chunk_and_embed, mock_get_file_mod_times):
        mock_get_file_mod_times.return_value = {'file1.txt': 1000}  # Simulate no changes in file modification time
        mock_chunk_and_embed.return_value = []  # No embedding happens in this case

        # Create a temporary directory to act as the document path
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary text file to simulate the documents
            temp_file_path = os.path.join(temp_dir, "file1.txt")
            with open(temp_file_path, 'w') as f:
                f.write("This is some test content.")

            # Create a temporary file for timestamp_file
            with tempfile.NamedTemporaryFile(delete=False) as temp_timestamp_file:
                temp_timestamp_file.write(b"{'file1.txt': 1000}")  # Write initial timestamp data
                temp_timestamp_file_path = temp_timestamp_file.name

            config = {
                'rag_settings': {
                    'document_path': temp_dir,  # Use the temp directory for documents
                    'timestamp_file': temp_timestamp_file_path,  # Use the temp file for the timestamp
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'chroma_path': 'test_chroma_path',
                    'embedding_model': 'text-embedding-3-large',
                    'separators': [" ", ",", "\n"]
                }
            }

            # Run function with the temporary directory and timestamp file
            with patch('os.path.exists', return_value=True):
                initialize_rag(config)

            # Assert that chunk_and_embed_documents was NOT called since there were no changes
            mock_chunk_and_embed.assert_not_called()

            # Assert that Chroma was NOT initialized
            mock_chroma.from_documents.assert_not_called()

            # Clean up the temporary file after the test
            os.remove(temp_timestamp_file_path)

    # @patch('rag.get_file_mod_times')
    # @patch('rag.chunk_and_embed_documents')
    # @patch('rag.Chroma')
    # def test_initialize_rag_with_changes(self, mock_chroma, mock_chunk_and_embed, mock_get_file_mod_times):
    #     mock_get_file_mod_times.return_value = {'file1.txt': 2000}  # Simulate file modification time has changed
    #     mock_chunk_and_embed.return_value = [{"text": "chunked content", "embedding": [0.1, 0.2]}]  # Mock some embedded documents

    #     # Create a temporary directory to act as the document path
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         # Create a temporary text file to simulate the documents
    #         temp_file_path = os.path.join(temp_dir, "file1.txt")
    #         with open(temp_file_path, 'w') as f:
    #             f.write("This is some test content.")

    #         # Create a temporary file for timestamp_file
    #         with tempfile.NamedTemporaryFile(delete=False) as temp_timestamp_file:
    #             temp_timestamp_file.write(b"{'file1.txt': 1000}")  # Write initial timestamp data
    #             temp_timestamp_file_path = temp_timestamp_file.name

    #         config = {
    #             'rag_settings': {
    #                 'document_path': temp_dir,  
    #                 'timestamp_file': temp_timestamp_file_path,  
    #                 'chunk_size': 1000,
    #                 'chunk_overlap': 200,
    #                 'chroma_path': 'test_chroma_path',
    #                 'embedding_model': 'text-embedding-3-large',
    #                 'separators': [" ", ",", "\n"]
    #             }
    #         }

    #         # Run the function
    #         with patch('os.path.exists', return_value=True):
    #             initialize_rag(config)

    #         # Assert that chunk_and_embed_documents was called since there were changes
    #         mock_chunk_and_embed.assert_called_once()

    #         # Assert that Chroma was initialized with the embedded documents
    #         mock_chroma.from_documents.assert_called_once()

    #         # Clean up the temporary file after the test
    #         os.remove(temp_timestamp_file_path)


class TestLoadRag(unittest.TestCase):
    @patch('rag.rag_ready', True)  # Patch the global rag_ready to True
    @patch('rag.Chroma')
    @patch('rag.OpenAIEmbeddings')
    def test_load_rag_when_ready(self, mock_openai_embeddings, mock_chroma):
        # Mock the embeddings and chroma instances
        mock_embedding_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embedding_instance
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        # Call the function with the mocks
        load_rag(embedding_model='text-embedding-3-large', chroma_path='test_chroma_path', openai_api_key='test_key')

        # Assert that OpenAIEmbeddings and Chroma were called when rag_ready is True
        mock_openai_embeddings.assert_called_once_with(openai_api_key='test_key', model='text-embedding-3-large')
        mock_chroma.assert_called_once_with(
            persist_directory='test_chroma_path',
            embedding_function=mock_embedding_instance,
            collection_name="ipcc-collection"
        )

    @patch('rag.rag_ready', False)  # Patch the global rag_ready to False
    @patch('rag.Chroma')
    @patch('rag.OpenAIEmbeddings')
    def test_load_rag_when_not_ready(self, mock_openai_embeddings, mock_chroma):
        # Call the function with the mocks
        load_rag(embedding_model='text-embedding-3-large', chroma_path='test_chroma_path', openai_api_key='test_key')

        # Assert that OpenAIEmbeddings and Chroma were NOT called when rag_ready is False
        mock_openai_embeddings.assert_not_called()
        mock_chroma.assert_not_called()
        

# Run the tests
if __name__ == '__main__':
    unittest.main()
