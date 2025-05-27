# Part of this software includes code from the "moabb" project, 
# licensed under the BSD 3-Clause License.

import logging
import os

import mne
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
import numpy as np
from scipy.io import loadmat

from datasets import BaseDataset
from helper_functions import download_file, file_mover, setup_logger


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class Cho2017(BaseDataset):
    '''Class to load the Cho et al. (2017) dataset'''
    
    BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/"

    def __init__(self, subjects=None, path=None):
        self.subjects = list(subjects) if subjects else list(range(1, 53)) # list(subjects) if subjects else istrint veliau
        self.path = path if path else os.path.join(os.getcwd(), 'data','raw', '1')

        super().__init__(    
            subjects=self.subjects,
            sessions_per_subject=1, 
            events = dict(left_hand=1, right_hand=2),
            code = "Cho2017 #1",
            interval = [0, 3],  # full trial is 0-3s, but edge effects
            doi = "10.5524/100295",
            additional_info = ""
        )
        
        self.log = setup_logger('Cho2017')

    def data_path(self, subject):
        filename = "s{:02d}.mat".format(subject)
        file_path = os.path.join(self.path, filename)

        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"File {file_path} does not exist")
    
    ## kažkas čia ne iki galo
    def _load_one_raw(self, subject):
        """Return data for a single subject."""
        fname = self.data_path(subject)

        data = loadmat(
            fname,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )["eeg"]

        # fmt: off
        eeg_ch_names = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ]
        # fmt: on
        emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
        ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
        ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
        montage = make_standard_montage("standard_1005")
        imagery_left = data.imagery_left - data.imagery_left.mean(axis=1, keepdims=True)
        imagery_right = data.imagery_right - data.imagery_right.mean(
            axis=1, keepdims=True
        )

        eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
        eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])

        # trials are already non continuous. edge artifact can appears but
        # are likely to be present during rest / inter-trial activity
        eeg_data = np.hstack(
            [eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r]
        )
        log.warning(
            "Trials demeaned and stacked with zero buffer to create "
            "continuous data -- edge effects present"
        )

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
        raw = RawArray(data=eeg_data, info=info, verbose=False)
        raw.set_montage(montage)

        return raw

    def _download_all_data(self):
        os.makedirs(self.path, exist_ok=True)

        for subject in self.subjects:
            filename = "s{:02d}.mat".format(subject)
            file_path = os.path.join(self.path, filename)
            url = f"{self.BASE_URL}mat_data/{filename}"
            if not os.path.exists(file_path):
                download_file(url, self.path, self.log)

        additional_files = [
            "readme.txt",
            "Questionnaire_results_of_52_subjects.xlsx",
            "trial_sequence.zip"
        ]

        for filename in additional_files:
            file_path = os.path.join(self.path, filename)
            url = f"{self.BASE_URL}{filename}"
            if not os.path.exists(file_path):
                download_file(url, self.path, self.log)

        log.info("All files have been downloaded.")

        file_mover(self.path, logger=self.log, filter_type='non_mat')
    
    def initialize(self):
        from helper_functions import save_raw_to_fif, file_mover
        self._download_all_data()
        for subject in self.subjects:
            
            try:
                raw = self._load_one_raw(subject)
            except OSError as e:
                self.log.error(f"Failed to load raw data for subject {subject}: {e}")
                continue  
            
            dataset_path = save_raw_to_fif(raw, dataset_no=1, file_name=f"s{subject}", logger=self.log)

        file_mover(self.path, logger=self.log, filter_type='non_mat', destination_dir=dataset_path)
        self.log.info("All files have been processed.")
        
    def load_one_raw_fif(self, subject):
        '''Read and return the raw data for a single subject'''
        path = os.path.join(os.getcwd(), 'data','raw_fif', '1', f's{subject}_raw.fif')

        try:
            raw = mne.io.read_raw_fif(path, preload=True)
        except FileNotFoundError as e:
            self.log.error(f"Failed to load raw data for subject {subject}: {e}")
            raw = None  
        
        return raw


    def load_fif_data(self, subjects=None):
        if subjects is None:
            subjects = self.subjects

        if isinstance(subjects, int):
            subjects = [subjects]

        data = {}
        for subject in subjects:
            raw = self.load_one_raw_fif(subject)
            data[subject] = raw
        return data