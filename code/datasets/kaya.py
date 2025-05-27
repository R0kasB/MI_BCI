import os
import glob
import requests
import mne
from scipy.io import loadmat
from mne.channels import make_standard_montage
from mne import create_info
import numpy as np
from mne.io import RawArray
from helper_functions import download_file

from datasets import BaseDataset
from helper_functions import setup_logger

class Kaya2018(BaseDataset):
    '''Class to load the Kaya2018 dataset
    
Nepatikrinta

Abbreviations used in the data record naming convention.

| Abbreviation       | Meaning                                      | Explanation                                                                                                  |
|--------------------|----------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| CLA                | Classical (CLA) left/right hand MI           | Paradigm#1                                                                                                   |
| HaLT               | Hand/leg/tongue (HaLT) MI                    | Paradigm#2                                                                                                   |
| FREEFORM           | Freestyle left/right hand MI                 | Paradigm#3                                                                                                   |
| 5F                 | 5 fingers (5F) MI                            | Paradigm#4                                                                                                   |
| NoMT               | No imagery, visual stimuli only              | Paradigm#5                                                                                                   |
| SGLHand            | Single hand                                  | Recording sessions in 5 F paradigm performed for the motor imagery of fingers on one hand                    |
| HFreq              | High frequency                               | Recording sessions recorded with 1000 Hz sampling rate setting in EEG 1200 as opposed to 200 Hz used in others|
| Tong               | Tongue                                       | Identifies recording sessions including tongue MI                                                            |
| St                 | State                                        | Indicates the total number of mental imageries present in the recording session                              |
| Inter              | Interface                                    | Indicates sessions using an interactive user interface with lower signal resolution and dynamic range        |
| LRHand             | Left/right hand                              | Indicates sessions for MI of left and right hand movements                                                   |
| LRHandLeg Tongue   | Left/right hand, leg and tongue              | Indicates sessions focusing on MI of left/right hand, left/right leg, and tongue movements                   |


Explanation of the numerical codes used in recording session interaction records

| "Marker" Code | Meaning            | "Marker" Code | Meaning        |
|---------------|--------------------|---------------|----------------|
| **CLA, HaLT, FreeForm, NoMT**      |**5F**         |                |
| 1             | Left hand          | 1             | Thumb          |
| 2             | Right hand         | 2             | Index finger   |
| 3             | Passive/neutral    | 3             | Middle finger  |
| 4             | Left leg           | 4             | Ring finger    |
| 5             | Tongue             | 5             | Pinkie finger  |
| 6             | Right leg          |               |                |
| 91            | Session break      |               |                |
| 92            | Experiment end     |               |                |
| 99            | Initial relaxation |               |                |


List of data files in the dataset grouped by recording session participant.

| Subject | Sex | Age group | Health condition | Prior BCI experience | BCI literacy level  | Data                                 |
|---------|-----|-----------|------------------|----------------------|---------------------|--------------------------------------|
| A       | M   | 20-25     | Healthy          | None                 | Intermediate-High    | 5F-SubjectA-160405, 5F-SubjectA-160408, CLA-SubjectA-160108, HaLT-SubjectA-160223, HaLT-SubjectA-160308, HaLT-SubjectA-160310 |
| B       | M   | 20-25     | Healthy          | None                 | Intermediate-Low     | 5F-SubjectB-151110, 5F-SubjectB-160309, 5F-SubjectB-160311, 5F-SubjectB-160316, CLA-SubjectB-151019, CLA-SubjectB-151020, CLA-SubjectB-151215, FREEFORM-SubjectB-151111, HaLT-SubjectB-160218, HaLT-SubjectB-160225, HaLT-SubjectB-160229 |
| C       | M   | 25-30     | Healthy          | None                 | Intermediate-High    | 5F-SubjectC-151204, 5F-SubjectC-160429, CLA-SubjectC-151126, CLA-SubjectC-151216, CLA-SubjectC-151223, FREEFORM-SubjectC-151208, FREEFORM-SubjectC-151210, HaLT-SubjectC-160224, HaLT-SubjectC-160302 |
| D       | M   | 25-30     | Healthy          | None                 | Intermediate-Low     | CLA-SubjectD-151125                 |
| E       | F   | 20-25     | Healthy          | None                 | Intermediate-Low     | 5F-SubjectE-160321, 5F-SubjectE-160415, 5F-SubjectE-160429, CLA-SubjectE-151225, CLA-SubjectE-160119, CLA-SubjectE-160122, HaLT-SubjectE-160219, HaLT-SubjectE-160226, HaLT-SubjectE-160304 |
| F       | M   | 30-35     | Healthy          | None                 | Intermediate-Low     | 5F-SubjectF-151027, 5F-SubjectF-160209, 5F-SubjectF-160210, CLA-SubjectF-150916, CLA-SubjectF-150917, CLA-SubjectF-150928, HaLT-SubjectF-160202, HaLT-SubjectF-160203, HaLT-SubjectF-160204, NoMT-SubjectF-160422 |
| G       | M   | 30-35     | Healthy          | None                 | Intermediate-High    | 5F-SubjectG-160413, 5F-SubjectG-160428, HaLT-SubjectG-160301, HaLT-SubjectG-160322, HaLT-SubjectG-160412 |
| H       | M   | 20-25     | Healthy          | None                 | Low                  | 5F-SubjectH-160804, HaLT-SubjectH-160720, HaLT-SubjectH-160722, NoMT-SubjectH-160628 |
| I       | F   | 25-30     | Healthy          | None                 | Low                  | 5F-SubjectI-160719, 5F-SubjectI-160723, HaLT-SubjectI-160609, HaLT-SubjectI-160628, NoMT-SubjectI-160512 |
| J       | F   | 20-25     | Healthy          | None                 | High                 | CLA-SubjectJ-170504, CLA-SubjectJ-170508, CLA-SubjectJ-170510, HaLT-SubjectJ-161121, NoMT-SubjectJ-161026 |
| K       | M   | 20-25     | Healthy          | None                 | Intermediate-Low     | HaLT-SubjectK-161027, HaLT-SubjectK-161108, NoMT-SubjectK-161025 |
| L       | F   | 20-25     | Healthy          | None                 | High                 | HaLT-SubjectL-161116, HaLT-SubjectL-161205, NoMT-SubjectL-161026 |
| M       | F   | 20-25     | Healthy          | None                 | Intermediate-High    | HaLT-SubjectM-161108, HaLT-SubjectM-161117, HaLT-SubjectM-161124, NoMT-SubjectM-161116 | 
'''
    
    def __init__(self, subjects=None, path=None, run=None, paradigm=None, hfreq=None):
        self.subjects = subjects if subjects else list(range(1,14))
        self.fiveF = dict(thumb=1, index=2, middle=3, ring=4, pinkie=5)
        self.subjects_info = ['SubjectA', 'SubjectB', 'SubjectC', 
                            'SubjectD', 'SubjectE', 'SubjectF', 
                            'SubjectG', 'SubjectH', 'SubjectI', 
                            'SsubjectJ', 'SubjectK', 'SubjectL', 
                            'SubjectM']
        self.path = path if path else os.path.join(os.getcwd(), 'data','raw', '3')

        self.paradigm = paradigm if paradigm else 'CLA'
        self.hfreq = hfreq if hfreq else False
        
        if self.paradigm == '5F':
            self.event_desc = {
                1: 'thumb',
                2: 'index_finger',
                3: 'middle_finger',
                4: 'ring_finger',
                5: 'pinkie_finger',
                6: 'right_leg',
                91: 'session_break',
                92: 'experiment_end',
                99: 'initial_relaxation'
            }            
        elif self.paradigm in ['CLA', 'HaLT', 'FreeForm', 'NoMT']:
            self.event_desc = {
                1: 'left_hand',
                2: 'right_hand',
                3: 'passive_state',
                4: 'left_leg',
                5: 'tongue',
                6: 'right_leg',
                91: 'session_break',
                92: 'experiment_end',
                99: 'initial_relaxation'
            }
        else:
            raise ValueError(f"Unknown paradigm {self.paradigm}")
        
        super().__init__(
            subjects=self.subjects,
            sessions_per_subject= ['CLA', 'HaLT', 'FREEFORM', 'NoMT','5F'],
            events=self.event_desc,
            code="Kaya2018 #3",
            interval=[],
            doi="",
            additional_info="5F interaction paradigm: 1-thumb MI, 2-index finger MI, 3-middle finger MI, 4-ring finger MI, 5-pinkie finger MI"
            )
        
        
        self.log = setup_logger('Kaya2018')
        
    
    def _load_one_raw(self, path):
        data = loadmat(path,squeeze_me=True,struct_as_record=False)['o']

        chnames=data.chnames.tolist()
        montage = make_standard_montage("standard_1005")
        sfreq = data.sampFreq
        info = create_info(
            ch_names=chnames + ['stim'], 
            ch_types=['eeg'] * len(chnames) + ['stim'], 
            sfreq=sfreq)

        data.data *= 1e-6 #mne.raw takes in data as V, not uV
        data_for_raw = np.vstack([data.data.T, data.marker])
        raw = RawArray(data=data_for_raw, info=info, verbose=True)
        
        # raw.drop_channels('X5') #idk ka daryt su tuo kanalu
        # Check if 'X3' is present in the data channels and drop it if so
        if 'X3' in raw.info['ch_names']:
            raw.drop_channels(['X3'])

        # Check if 'X5' is present in the data channels and drop it if so
        if 'X5' in raw.info['ch_names']:
            raw.drop_channels(['X5'])
            
        raw.set_montage(montage)

        # Find the events
        events = mne.find_events(raw, stim_channel='stim', min_duration=1)
        annotations = mne.annotations_from_events(events=events, sfreq=sfreq)

        onsets = events[:, 0] / sfreq  # Onsets in seconds
        descriptions = [self.event_desc[e] for e in events[:, 2]]

        # Calculate offsets (next onset or end of recording)
        # durations = np.diff(np.append(onsets, raw.times[-1]))

        # Create annotations with the correct duration
        annotations = mne.Annotations(onset=onsets, duration=[1] * len(onsets), description=descriptions)
        
        raw.set_annotations(annotations)        
        return raw
    
    # Base URL of the Figshare API for the collection
    collection_url = 'https://api.figshare.com/v2/collections/3917698/articles'

    # Function to get all dataset URLs from the collection page using Figshare API with pagination
    def _get_all_dataset_urls(self, api_url):
        
        dataset_urls = []
        page = 1
        while True:
            response = requests.get(api_url, params={'page': page, 'page_size': 100})
            datasets = response.json()
            if not datasets:
                break
            for dataset in datasets:
                dataset_id = dataset['id']
                dataset_url = f"https://api.figshare.com/v2/articles/{dataset_id}/files"
                dataset_urls.append(dataset_url)
            page += 1
        
        self.log.info(f"Dataset URLs found: {len(dataset_urls)} datasets.")  # Debug print
        return dataset_urls

    # Function to get all file download links and their names from a dataset page using Figshare API
    def get_download_links(self, api_url):
        response = requests.get(api_url)
        files = response.json()
        
        download_links = [(file['download_url'], file['name']) for file in files]
        self.log.info(f"Download links found on page {api_url}: {download_links}")  # Debug print
        return download_links

    def _download_all_data(self):
        # Define the download directory
        Kaya2018_path = self.path

        # Ensure the directory exists
        os.makedirs(Kaya2018_path, exist_ok=True)
        
        # Fetch all dataset URLs from the collection page with pagination
        dataset_urls = self._get_all_dataset_urls(self.collection_url)

        # Collect all file download links from each dataset page
        all_file_links = []
        for dataset_url in dataset_urls:
            download_links = self.get_download_links(dataset_url)
            all_file_links.extend(download_links)

        # List of download URLs (assuming `all_file_links` is defined)
        download_links = all_file_links

        # Download each file with its original name
        for file_url, file_name in all_file_links:
            self.log.info(f"Downloading {file_url}")
            download_file(file_url, Kaya2018_path, self.log, name=file_name)
            
        self.log.info("All datasets downloaded.")
        
        
    def initialize(self):
        from helper_functions import save_raw_to_fif, find_unprocesed_paths, file_mover
        
        self._download_all_data()
        file_mover(self.path, self.log, filter_type='non_mat', destination_folder_name='additional_files')
        
        # Initialize the file processing counter
        files_processed_no = 0

        # Find unprocessed paths
        paths_hfreq, info = find_unprocesed_paths(self.path, marks='HFREQ.mat', info=True)
        paths_lowfreq = info['unfiltered_paths_list']
        total_paths_no = info['Nr_paths']
        data_destination_dir = os.path.join(os.getcwd(), 'data', 'raw_fif', '3')
        
        # Process files based on the paradigm and subject
        for mark in ['CLA', 'HaLT', 'FREEFORM', 'NoMT', '5F', '5F_HFREQ']:  # 2nd level
            if mark != '5F_HFREQ':
                paradigm_paths = find_unprocesed_paths(path_list=paths_lowfreq, marks=mark)
            else:
                paradigm_paths = paths_hfreq

            for subject_no, subject in enumerate(self.subjects_info, start=1):  # 3rd level
                subject_paths = find_unprocesed_paths(path_list=paradigm_paths, marks=subject)
                
                if not subject_paths:
                    self.log.warning(f"No unprocessed paths found for subject {subject} in {mark} paradigm. Skipping subject.")
                    continue

                for run_no, path in enumerate(subject_paths):  # 4th level
                    # Create the file name using the sequence number
                    file_name = f"s{subject_no}.{run_no}"
                    compound_dir = os.path.join(data_destination_dir, mark)
                    
                    try:
                        # Load the raw data and save it to FIF format
                        save_raw_to_fif(self._load_one_raw(path), dataset_no=3, file_name=file_name, optional_dir=compound_dir, logger=self.log)
                        files_processed_no += 1  # Increment counter only after successful processing
                    except Exception as e:
                        self.log.error(f"Failed to process file {file_name} at {path}. Error: {e}")

        # Check if all files were processed
        if files_processed_no == total_paths_no:
            self.log.info("All files have been processed.")
        else:
            self.log.error(f"Some files were not processed. Files processed: {files_processed_no}, Total files: {total_paths_no}")
            
        file_mover(folder=self.path,filter_type='non_mat', destination_dir=data_destination_dir, logger=self.log, copy=True)       
    
    def load_one_raw_fif(self, subject, paradigm, run):
        '''Read and return the raw data for a single subject'''
        
        path = os.path.join(os.getcwd(), 'data','raw_fif', '3', paradigm, f's{subject}.{run}_raw.fif')

        try:
            raw = mne.io.read_raw_fif(path, preload=True)
        except FileNotFoundError as e:
            self.log.error(f"Failed to load raw data for subject {subject}: {e}")
            raw = None  # You may choose to return None or handle it differently
        
        return raw


    def load_fif_data(self, subjects=None, paradigms=None, runs=None):
        if subjects is None:
            subjects = self.subjects
        if paradigms is None:
            paradigms = ['CLA', 'HaLT', 'FREEFORM', 'NoMT','5F', '5F_HFREQ']
            
        if isinstance(subjects, int):
            subjects = [subjects]
        if isinstance(paradigms, str):
            paradigms = [paradigms]


        data = {}
        for paradigm in paradigms:
            data[paradigm] = {}
            for subject in subjects:
                current_runs = runs
                if current_runs is None:
                    subject_paths = glob.glob(os.path.join(os.getcwd(), 'data','raw_fif', '3', paradigm, f's{subject}.*_raw.fif'))
                    run_no = len(subject_paths)
                    
                    if run_no in {1, 2, 3}:  # Check if it's within the expected range
                        current_runs = list(range(0, run_no))
                    else:
                        self.log.error(f"Found {len(subject_paths)} files for subject {subject}. Expected 1, 2, or 3 files.")
                        continue  # Skip this subject if the number of runs is unexpected
                data[paradigm][subject] = {}
                for run in current_runs:
                    raw = self.load_one_raw_fif(subject, run=run, paradigm=paradigm)
                    data[paradigm][subject][run] = raw
        return data