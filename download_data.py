import requests
import tarfile
import zipfile
import os
import shutil
import sys
import yaml 
import argparse

def download_file(url, local_filename):
    """Attempt to download a file from a URL and save it locally, with a progress indicator."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            # Get total file size from headers, if available
            total_length = r.headers.get('content-length')
            if total_length is not None:
                total_length = int(total_length)
                downloaded = 0

            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if total_length is not None:
                        downloaded += len(chunk)
                        # Calculate the percentage of the file downloaded and update the progress bar
                        done_percentage = int(100 * downloaded / total_length)
                        # Update the progress bar
                        sys.stdout.write(f"\rDownloading {local_filename}: {done_percentage}%")
                        sys.stdout.flush()
            if total_length is not None:
                sys.stdout.write('\n')  # Move the cursor to the next line after download completes

        return True
    except requests.RequestException as e:
        print(f"\033[93mWarning: Failed to download {url}. Please download manually.\033[0m")
        print(f"\033[91mError: {e}\033[0m")
        return False


def extract_tar(file_path, extract_to='.'):
    """Extract tar file and handle errors."""
    try:
        with tarfile.open(file_path) as tar:
            tar.extractall(path=extract_to)
        os.remove(file_path)
    except Exception as e:
        print(f"\033[93mWarning: Failed to extract {file_path}.\033[0m")

def extract_zip(file_path, extract_to='.'):
    """Extract zip file and handle errors."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(file_path)
    except Exception as e:
        print(f"\033[93mWarning: Failed to extract {file_path}.\033[0m")

def extract_arch(file_path, extract_to='.', archive_type=''):
    """Extract tar/zip file and handle errors.
        if archive_type='' (default) file will be moved
    """
    if not archive_type:
        _, file_extension = os.path.splitext(file_path)
        if file_extension in ['.zip']:
            archive_type = 'zip'
        elif file_extension in ['.tar']:
            archive_type = 'tar'
    try:
        if archive_type=='zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_type=='tar':
            with tarfile.open(file_path) as tar:
                tar.extractall(path=extract_to)
        else:
            #cp file to     
            destination_file = os.path.join(extract_to, os.path.basename(file_path))   
            shutil.copyfile(file_path,destination_file)    
        os.remove(file_path)
    except Exception as e:
        print(f"\033[93mWarning: Failed to extract {file_path}.\033[0m")        

def create_dir(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def remove_dir(path):
    """Remove a directory if it exists."""
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

def main():
    # Parse command-line argument (--source_files)
    parser = argparse.ArgumentParser(description="Download and extract the raw source files of the RAG.")
    parser.add_argument('--source_files', type=bool, default=False, help='Whether to download and extract source files (IPCC text reports).')
    args = parser.parse_args()

    # Load the YAML file
    with open('data_sources.yml', 'r') as file:
        data_config = yaml.safe_load(file)

    base_path = data_config['base_path']
    sources = data_config['sources']

    #make subdirs list and clean it
    subdirs = []
    for entry in sources:
        subdirs.append(entry['subdir'])
    subdirs = set(subdirs)
    subdirs = list(subdirs)
    subdirs = [folder for folder in subdirs if folder not in ['.', './']]

    for subdir in subdirs:
        create_dir(os.path.join(base_path, subdir))

    # Download and extract files
    
    files_downloaded = []
    files_skiped = []
    urls_skiped = []
    subdirs_skiped = []

    for entry in sources:
        file = entry['filename']
        url  = entry['url']
        subdir = os.path.join(base_path, entry['subdir'])

        # Skip downloading source files of RAG unless --source_files is set to True
        if file == 'ipcc_text_reports.zip' and not args.source_files:
            print("Skipping IPCC text report download as --source_files flag is not set or False.")
            continue

        if download_file(url, file):
            extract_arch(file, subdir)
            files_downloaded.append(file)        
        else:
            files_skiped.append(file)
            urls_skiped.append(url)
            subdirs_skiped.append(subdir)
 
    if (files_skiped):
        print('\n')                      
        print('----------------------------------------------')                      
        print(f"\033[91mFiles not downloaded, please download manualy:\033[0m")
        for i,file in enumerate(files_skiped):
            print('--------')               
            print(f"\033[93mFile:\033[0m", file)
            print(f"\033[93mUrl:\033[0m", urls_skiped[i])        
            print(f"\033[93munpack it into the:\033[0m ", subdirs_skiped[i])            
            print('--------')        

    # I would leave it for a while
    #print('\n')                      
    #print('----------------------------------------------')                      
    #print("You also need to download the natural hazard data (for which you have to create a free account). Please download the CSV - Disaster Location Centroids [zip file] and unpack it into the 'data/natural_hazards' folder. Your file should automatically be called 'pend-gdis-1960-2018-disasterlocations.csv'. If not, please change the file name accordingly.")
    #print(f"\033[93mhttps://sedac.ciesin.columbia.edu/data/set/pend-gdis-1960-2018/data-download\033[0m")
    #print('-------------------')                      
    
if __name__ == "__main__":
    main()
