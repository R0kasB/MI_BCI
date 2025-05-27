import os
import logging
from urllib.parse import  urljoin
from datasets import BaseDataset
from helper_functions import download_file, setup_logger

import mne
from scipy.io import loadmat
from mne import Annotations
from mne.io import RawArray

class BCI4_1(BaseDataset):
    '''Class to load the BCI Competition IV dataset 1'''
    def __init__(self, subjects=None, path=None):
        self.subjects=subjects if subjects else list(range(1, 8))
        self.subjects_codes = ['a','b','c','d','e','f','g']
        self.path = path if path else os.path.join(os.getcwd(), 'data','raw', '4')
        self.test_type = ['calibration', 'evaluation']
        
        super().__init__(
            subjects = self.subjects,
            sessions_per_subject = [], 
            events = [],
            code = 'BCI4_1 #4',
            interval = [],
            doi = '',
            additional_info = ''
        )
            
        # Configure logging
        self.log = setup_logger('BCI4_1')

    def load_one_raw_fif(self, session, subject):
        '''Read and return the raw data for a single subject'''
        path = os.path.join(os.getcwd(), 'data','raw_fif', '4',session, f's{subject}_raw.fif')
        raw = mne.io.read_raw_fif(path, preload=True)
        return raw  

    def load_fif_data(self, subjects=None, sessions=None):
        if subjects is None:
            subjects = self.subjects
        if sessions is None:
            sessions = self.test_type
        
        if isinstance(sessions, str):
            sessions = [sessions]
            
        if isinstance(subjects, int):
            subjects = [subjects]
        
        data = {}
        for session in sessions:
            data[session] = {} 
            for subject in subjects:
                raw = self.load_one_raw_fif(session,subject)
                data[session][subject] = raw
        return data

    def _load_one_raw(self, path):
        data_loaded = loadmat(path, squeeze_me=True,struct_as_record=False)
        
        chan_names = data_loaded['nfo'].clab.tolist()
        chan_x_pos = data_loaded['nfo'].xpos
        chan_y_pos = data_loaded['nfo'].ypos
        sfreq = data_loaded['nfo'].fs
        data = data_loaded['cnt'].T
        # data = data.astype(np.float64)  # Convert to double precision (equivalent to MATLAB's double)
        # data = 0.1 * data  # Scale the values to convert them to microvolts

        info = mne.create_info(ch_names = chan_names, ch_types=['eeg'] * len(chan_names), sfreq=sfreq)
        raw = RawArray(data=data, info=info, verbose=True)

        # Initialize an empty dictionary for the electrode coordinates
        electrode_coordinates = {}

        # Loop through each channel and create a dictionary entry with a z position of 0
        for i, chan in enumerate(chan_names):
            electrode_coordinates[chan] = [chan_x_pos[i], chan_y_pos[i], 0.0]

        montage = mne.channels.make_dig_montage(ch_pos=electrode_coordinates, coord_frame='head')
        raw.set_montage(montage)

        '''
        # Assuming `cnt` is your continuous EEG data as a NumPy array
        data = data.astype(np.float64)  # Convert to double precision (equivalent to MATLAB's double)
        data = 0.1 * data  # Scale the values to convert them to microvolts
        #pabandyt scalint veliau
        '''
        
        try:
            cue_positions = data_loaded['mrk'].pos
            cue_classes = data_loaded['mrk'].y
        except KeyError:
            self.log.error("The marker ('mrk') data is missing in the loaded data. Returning raw data without annotations.")
            return raw  # or handle the missing data appropriately

        # Convert sample positions to times (assuming you have the sampling frequency)
        cue_times = cue_positions / sfreq  # Convert positions to times in seconds

        # Define event descriptions based on cue classes
        descriptions = ['Class 1' if y == -1 else 'Class 2' for y in cue_classes]

        # Define durations for each event
        is_calibration_data = True  # Set this to False if it's evaluation data

        if is_calibration_data:
            durations = [4.0] * len(cue_times)  # Fixed 4 seconds duration for each cue during calibration
        else:
            # patikrinti trukmes
            pass

        # Create MNE Annotations with the specified durations
        annotations = Annotations(onset=cue_times, duration=durations, description=descriptions)

        # Add annotations to raw data
        raw.set_annotations(annotations)
        return raw
    
    def _download_all_data(self):
        # Define the download directory
        BCI4_1_path = self.path

        # Ensure the directory exists
        os.makedirs(BCI4_1_path, exist_ok=True)

        calibration_url = 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1calib_1000Hz_mat.zip'
        evaluation_url = 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1eval_1000Hz_mat.zip'

        # Download the files
        download_file(calibration_url, BCI4_1_path, self.log)
        download_file(evaluation_url, BCI4_1_path, self.log)
        
    def initialize(self):
        from helper_functions import find_unprocesed_paths, save_raw_to_fif
        files_processed = 0
        self._download_all_data()
        for mark, name in zip(['calib', 'eval'], ['calibration', 'evaluation']):
            paths, info = find_unprocesed_paths(self.path, marks = mark, info=True)
            
            total_paths = info['Nr_paths']
            files_processed += info['Nr_filtered_paths']
            
            for i, path in enumerate(paths, start=1):
                # Create the file name using the sequence number
                file_name = f"s{i}"
                
                # Load the raw data and save it to FIF format
                save_raw_to_fif(self._load_one_raw(path), dataset_no=4, file_name=file_name, optional_dir=name, logger=self.log)

        if files_processed == total_paths:
            self.log.info("All files have been processed.")
        else:
            self.log.error(f"Some files were not processed., Files processed: {files_processed}, Total files: {total_paths}")








class BCI4_2a(BaseDataset):
    '''Class to load the BCI Competition IV dataset 2a'''

    def __init__(self, subjects=None, path=None):
        self.subjects = subjects if subjects else list(range(1, 10))
        self.path = path if path else os.path.join(os.getcwd(), 'data', '5')
        self.subjects_codes = ['A01T', 'A01E', 'A02T', 'A02E', 'A03T', 'A03E', 'A04T', 'A04E', 'A05T', 'A05E', 'A06T', 'A06E', 'A07T', 'A07E', 'A08T', 'A08E', 'A09T', 'A09E']
        
        super().__init__(
            subjects = self.subjects,
            sessions_per_subject = list(range(0, 9)),
            events = ['left hand', 'right hand', 'feet', 'tongue'],
            code = 'BCI4_2a #5',
            interval = [],
            doi = '',
            additional_info = ''
        )

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(self.__class__.__name__)

    def download_all_data(self):
        # Define the download directory
        BCI4_2a_path = self.path

        # Ensure the directory exists
        os.makedirs(BCI4_2a_path, exist_ok=True)

        # Example URLs for BCI4_2a dataset
        base_url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2014/'
        pdf_url = 'http://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf'

        # Use the download_file method from BCI4_1 class
        for subject in self.subjects_codes:
            url = urljoin(base_url, subject + '.mat')
            download_file(url, BCI4_2a_path, self.log)
        download_file(pdf_url, BCI4_2a_path, self.log)
    