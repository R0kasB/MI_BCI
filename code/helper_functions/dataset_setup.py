import glob
import os
# import regex as re
import re
import requests
from tqdm import tqdm
import zipfile
from urllib.parse import urlparse

import shutil

def find_unprocesed_paths(path_to_data_dir=None, marks=(list,str), file_type = '.mat', info = False, path_list=None, by_marks=True):
    """
    Identifies files in a specified directory (or from a given list of paths) that either contain or exclude specific marks (strings) in their filenames.
    The function can filter based on whether the marks are present or absent. It also allows for the option to return detailed information about the filtering process.

    Parameters:
        path_to_data_dir (str, optional): The directory path containing the files to process.
        marks (str or list of str): A string or list of strings that the filenames should contain or exclude.
        file_type (str, optional): The file extension to filter by (default is '.mat').
        info (bool, optional): If set to True, returns additional information about the filtering process.
        path_list (list of str, optional): A list of file paths to process instead of a directory.
        by_marks (bool, optional): If True, filter files that contain the marks; if False, filter files that do not contain the marks.

    Returns:
        list: If `info` is False, returns a list of filtered file paths.
        tuple: If `info` is True, returns a tuple containing the filtered paths and a dictionary with additional information about the filtering process.
    desc By chatGPT
    """
    if isinstance(marks, str):
        marks = [marks]
    if path_to_data_dir:
        paths = glob.glob(os.path.join(path_to_data_dir, '*' + file_type))
    elif path_list:
        paths = path_list
        
    # Get paths from directory if not provided
    if path_to_data_dir is not None:
        paths = glob.glob(os.path.join(path_to_data_dir, '*' + file_type))
    elif path_list is not None:
        paths = path_list
    else:
        # Raise an error if neither path_to_data_dir nor path_list is provided
        raise ValueError("Either path_to_data_dir or path_list must be provided.")

    if by_marks:
        filtered_paths = [path for path in paths if all(re.search(mark, os.path.basename(path)) for mark in marks)]
    else:
        filtered_paths = [path for path in paths if not any(re.search(mark, os.path.basename(path)) for mark in marks)]

    if info:
        unfiltered_paths = [path for path in paths if path not in filtered_paths]
        return filtered_paths, {'Nr_paths' : len(paths), 
                                'Nr_filtered_paths' : len(filtered_paths), 
                                'all_paths_list' : paths,
                                'unfiltered_paths_list' : unfiltered_paths}
    return filtered_paths

def download_file(url, folder, logger, delete_zip=True, name=None):
    """
    Downloads a file from a given URL to a specified folder. It supports displaying a progress bar during the download, handles zip files by extracting them if needed, 
    and provides options to delete the zip file after extraction. The function checks if the file already exists before downloading it.

    Parameters:
        url (str): The URL of the file to download.
        folder (str): The folder where the downloaded file should be saved.
        logger (logging.Logger): A logger to record the download process.
        delete_zip (bool, optional): If True, deletes the zip file after extraction (default is True).
        name (str, optional): A custom name for the downloaded file. If not provided, the original file name from the URL is used.

    Returns:
        None. The function logs the download process and extracts files if necessary.
    desc By chatGPT
    """
    # Extract the original file name from the URL if name is not provided
    original_filename = os.path.basename(urlparse(url).path)
    local_filename = os.path.join(folder, name if name else original_filename)

    # Check if the file already exists
    if os.path.exists(local_filename):
        logger.info(f"File {local_filename} already exists. Skipping download.")
        return

    try:
        logger.info(f"Starting download of {original_filename} to {folder}")
        
        # Streaming, so we can iterate over the response
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Total size in bytes.
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        # Initialize the tqdm progress bar
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{original_filename}", ascii=True) as tqdm_bar:    
            with open(local_filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    tqdm_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and tqdm_bar.n != total_size:
            logger.error(f"ERROR: Incomplete download for {local_filename}. Expected {total_size}, got {tqdm_bar.n}")
        else:
            logger.info(f"{original_filename} has ben dowloaded")

            # Check if the downloaded file is a zip file and unzip it
            if local_filename.endswith('.zip'):
                with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                    zip_ref.extractall(folder)
                logger.info(f"Extracted {local_filename} to {folder}")

                if delete_zip:
                    # Delete the zip file after extraction
                    os.remove(local_filename)
                    logger.info(f"Deleted the zip file: {local_filename}")

    except requests.exceptions.RequestException as e:
        logger.error(f"ERROR: Failed to download {url}. Reason: {e}")

def file_mover(folder, logger, filter_type=None, filter_value=None, destination_dir=None, destination_folder_name='additional_files', copy=False):
    """
    Move or copy files from the given folder to a specified destination directory based on a selected filter type.
    
    Parameters:
        folder (str): The source folder from which to move or copy files.
        logger (logging.Logger): Logger to log the process.
        filter_type (str): Type of filter to apply ('non_mat', 'extension', 'size').
        filter_value: The value associated with the filter (e.g., list of extensions, size in KB).
        destination_dir (str): The directory where the files should be moved or copied.
        destination_folder_name (str): Name of the subfolder within the destination directory to move or copy files into. Defaults to 'additional_files'.
        copy (bool): If True, files will be copied instead of moved. Defaults to False.
    
    Example Usage:

    # Move all non-mat files to a specific directory
    file_mover(folder, logger, filter_type='non_mat', destination_dir='/path/to/destination')

    # Copy all non-mat files to a specific directory within a subfolder
    file_mover(folder, logger, filter_type='non_mat', destination_dir='/path/to/destination', copy=True)

    # Move files with specific extensions to a specific directory
    file_mover(folder, logger, filter_type='extension', filter_value=['.txt', '.csv'], destination_dir='/path/to/destination')

    # Copy files larger than a specific size to a specific directory within a subfolder
    file_mover(folder, logger, filter_type='size', filter_value=500, destination_dir='/path/to/destination', copy=True)
    desc By chatGPT
    """
    # Determine the final destination directory
    if destination_dir:
        destination_folder = os.path.join(destination_dir, destination_folder_name)
    else:
        destination_folder = os.path.join(folder, destination_folder_name)

    os.makedirs(destination_folder, exist_ok=True)

    # Define filter functions
    def filter_non_mat_files(filename):
        return not filename.endswith('.mat')
    
    def filter_by_extension(filename):
        return any(filename.endswith(ext) for ext in filter_value)
    
    def filter_by_size(filename):
        file_path = os.path.join(folder, filename)
        return os.path.getsize(file_path) > filter_value * 1024

    # Select the appropriate filter function
    if filter_type == 'non_mat':
        filter_func = filter_non_mat_files
    elif filter_type == 'extension' and filter_value:
        filter_func = filter_by_extension
    elif filter_type == 'size' and filter_value:
        filter_func = filter_by_size
    else:
        filter_func = None  # If no valid filter is provided, process all files

    for root, _, files in os.walk(folder):
        for file in files:
            if filter_func is None or filter_func(file):
                source = os.path.join(root, file)
                destination = os.path.join(destination_folder, file)
                
                if copy:
                    shutil.copy2(source, destination)
                    logger.info(f"Copied {file} to {destination_folder}")
                else:
                    shutil.move(source, destination)
                    logger.info(f"Moved {file} to {destination_folder}")


def save_raw_to_fif(raw, dataset_no, file_name, optional_dir=None, logger=None):
    """
    Saves raw data into a FIF format file in a specified directory. The function supports creating directories if they don't exist and allows for the use of 
    an optional subdirectory within the main dataset path. It also logs the process and handles potential errors related to the raw data format.

    Parameters:
        raw (object): The raw data object to be saved.
        dataset_no (int): The dataset number, used to create a directory for saving the FIF file.
        file_name (str): The name of the output FIF file (without the extension).
        optional_dir (str, optional): An additional directory within the main dataset path where the file should be saved.
        logger (logging.Logger, optional): A logger to record the saving process.

    Returns:
        None. The function logs the saving process and handles errors if the raw data is not in the expected format.
    desc By chatGPT
    """
    dataset_path = os.path.join(os.getcwd(), 'data', 'raw_fif', str(dataset_no))
    
    os.makedirs(dataset_path, exist_ok=True)
        # If optional_dir is provided, include it in the save path
    if optional_dir is not None:
        save_path = os.path.join(dataset_path, optional_dir)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = dataset_path
    
    try:
        raw.save(os.path.join(save_path, file_name + '_raw.fif'), overwrite=True)
    except AttributeError:
        if logger is not None:
            logger.error("The raw data provided does not have the expected structure or is missing. Cannot save the file.")
        return None  # Handle the missing data appropriately
        
    # Check if the file already exists
    if logger is not None:
        logger.info(f"{file_name} saved as {file_name}.fif in {save_path}")
    return dataset_path