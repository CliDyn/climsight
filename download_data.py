import requests
import tarfile
import zipfile
import os
import shutil
import sys

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

def create_dir(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def remove_dir(path):
    """Remove a directory if it exists."""
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

def main():
    base_path = './data'

    # Download and extract files
    
    ## Download firts main file 
    files_downloaded = []

    file = 'data_climate_foresight.tar'
    url  = 'https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/clisight/data_climate_foresight.tar'
    if download_file(url, file):
        extract_tar(file)
        files_downloaded.append(file)        

    ## Delete folders and files that we do not need !!!! this should be resolved by removing these files in archive
    
    # Remove duplicates directory
    remove_dir(os.path.join(base_path, 'natural_earth'))

    # Create necessary directories
    subdirs = ['natural_earth/coastlines',
               'natural_earth/land',
               'natural_earth/rivers',
               'natural_earth/lakes',
               'natural_hazards',
               'population']
    
    for subdir in subdirs:
        create_dir(os.path.join(base_path, subdir))

    ## Download and extract other files
    ###  -------------  coastline
    file = 'ne_10m_coastline.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/coastlines')
        files_downloaded.append(file)

    ###  -------------  land
    file = 'ne_10m_land.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/land')
        files_downloaded.append(file)

    ###  -------------  rivers
    file = 'ne_10m_rivers_lake_centerlines.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/rivers')
        files_downloaded.append(file)    
        
    file = 'ne_10m_rivers_australia.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_australia.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/rivers')
        files_downloaded.append(file)    

    file = 'ne_10m_rivers_europe.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_europe.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/rivers')
        files_downloaded.append(file)    

    file = 'ne_10m_rivers_north_america.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_north_america.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/rivers')
        files_downloaded.append(file)    
    ###  -------------  Lakes
    file = 'ne_10m_lakes.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/lakes')
        files_downloaded.append(file)    

    file = 'ne_10m_lakes_australia.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_australia.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/lakes')
        files_downloaded.append(file)    

    file = 'ne_10m_lakes_europe.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_europe.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/lakes')
        files_downloaded.append(file)    

    file = 'ne_10m_lakes_north_america.zip'
    url  = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_north_america.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/natural_earth/lakes')
        files_downloaded.append(file)    
    ###  -------------  popolation
    file = 'WPP2022_Demographic_Indicators_Medium.zip'
    url  = 'https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2022_Demographic_Indicators_Medium.zip'
    if download_file(url, file):
        extract_zip(file, base_path+'/population')
        files_downloaded.append(file)    

    ## Add similar blocks for other files
    #file = ''
    #url  = ''
    #if download_file(url, file):
    #    extract_zip(file, base_path+'/natural_earth/coastlines/lakes')
    #    files_downloaded.append(file)    
    

    # Check downloaded files
    print("Files downloaded successfully:")
    for file in files_downloaded:
        print(file)

if __name__ == "__main__":
    main()
