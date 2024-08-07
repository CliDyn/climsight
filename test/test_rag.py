import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock

# Append the directory containing the module to sys.path
module_dir = os.path.abspath('../src/climsight/')
if module_dir not in sys.path:
    sys.path.append(module_dir)

# Import module functions
from rag import (
    get_file_mod_times,
    get_file_names,
    load_docs,
    split_docs,
    initialize_rag_database
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


class TestInitializeRAGDatabase(unittest.TestCase):
    @unittest.skipIf(
        "OPENAI_API_KEY" not in os.environ,
        "Skipping test because OPENAI_API_KEY is not set in the environment."
    )
    @patch('rag.get_file_mod_times')
    @patch('rag.get_file_names')
    @patch('rag.load_docs')
    @patch('rag.split_docs')
    @patch('rag.OpenAIEmbeddings')
    @patch('rag.Chroma')
    def test_initialize_rag_database_no_changes(self, mock_chroma, mock_embeddings, mock_split_docs, mock_load_docs, mock_get_file_names, mock_get_file_mod_times):
        # Mock return values
        mock_get_file_mod_times.return_value = {'file1.txt': 1000} # random number for when file has been changed
        mock_get_file_names.return_value = ['file1.txt']
        mock_load_docs.return_value = [Document(page_content="Test content")]
        mock_split_docs.return_value = [Document(page_content="Test content chunk")]

        # Create a temporary timestamp file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_timestamp_file:
            temp_data = "{'file1.txt': 1000}"  # Using a string to simulate the file contents; this has to be the same number to simulate "no change"
            temp_timestamp_file.write(temp_data)
            temp_timestamp_file_path = temp_timestamp_file.name

        # Log the path and contents of the temporary file
        print(f"Temporary timestamp file path: {temp_timestamp_file_path}")

        # Read and log the content of the temp file to ensure it's correct
        with open(temp_timestamp_file_path, 'r') as temp_file:
            content = temp_file.read()
            print(f"Temporary file content before test: {content}")

        # Patch the path to use the temporary file
        original_timestamp_file = initialize_rag_database.__globals__['timestamp_file']
        initialize_rag_database.__globals__['timestamp_file'] = temp_timestamp_file_path

        try:
            # Run the database initialization
            initialize_rag_database()

            # Read the content of the timestamp file after initialization
            with open(temp_timestamp_file_path, 'r') as temp_file:
                new_content = temp_file.read()
                print(f"Temporary file content after test: {new_content}")

            # Manually parse the content for comparison
            last_mod_times = eval(new_content) # Using eval should be ok since it's controlled input
            current_mod_times = {'file1.txt': 1000}  # This should match

            # Log comparison
            print(f"Expected mod times: {current_mod_times}")
            print(f"Actual mod times read: {last_mod_times}")

            # Assert that the database was not updated
            mock_chroma.from_documents.assert_not_called()

        finally:
            # Restore the original timestamp file path
            initialize_rag_database.__globals__['timestamp_file'] = original_timestamp_file

            # Clean up
            os.remove(temp_timestamp_file_path)

    @unittest.skipIf(
        "OPENAI_API_KEY" not in os.environ,
        "Skipping test because OPENAI_API_KEY is not set in the environment."
    )
    @patch('rag.get_file_mod_times')
    @patch('rag.get_file_names')
    @patch('rag.load_docs')
    @patch('rag.split_docs')
    @patch('rag.OpenAIEmbeddings')
    @patch('rag.Chroma')
    def test_initialize_rag_database_with_changes(self, mock_chroma, mock_embeddings, mock_split_docs, mock_load_docs, mock_get_file_names, mock_get_file_mod_times):
        # Mock return values
        mock_get_file_mod_times.return_value = {'file1.txt': 2000}
        mock_get_file_names.return_value = ['file1.txt']
        mock_load_docs.return_value = [Document(page_content="Test content")]
        mock_split_docs.return_value = [Document(page_content="Test content chunk")]

        # Create a temporary timestamp file with old modification time
        with tempfile.NamedTemporaryFile(delete=False) as temp_timestamp_file:
            temp_timestamp_file.write(b"{'file1.txt': 1000}")
            temp_timestamp_file_path = temp_timestamp_file.name

        # Run the database initialization
        initialize_rag_database(force_update=True)

        # Assert that the database was updated
        mock_chroma.from_documents.assert_called_once()

        # Clean up
        os.remove(temp_timestamp_file_path)


# Run the tests
if __name__ == '__main__':
    unittest.main()
