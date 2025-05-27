import os
import glob
import requests
import mne
from scipy.io import loadmat
from mne.channels import make_standard_montage
from mne import create_info
import numpy as np
from mne.io import RawArray


from datasets import BaseDataset
from helper_functions import setup_logger

from functools import partialmethod

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat



Lee2019_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"

aditional_url = ["https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/Questionnaire_results.csv",
                    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/random_cell_order.mat",
                    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/readme.txt"
                    ]

class Lee2019(BaseDataset):

    def __init__(self, subjects=None, path=None, paradigm=None, sessions=None,
                 train_run=True, test_run=None, resting_state=False):
        
        self.paradigm = paradigm if paradigm else "MI"
        self.path = path if path else os.path.join(os.getcwd(), 'data','raw', '20')
        os.makedirs(self.path, exist_ok=True)

        self.sessions = sessions if sessions else (1, 2)
        self.subjects = subjects if subjects else list(range(1,55))
        if self.paradigm == "MI":
            interval = [
                0.0,
                4.0,
            ]  # [1.0, 3.5] is the interval used in paper for online prediction
                # 0.00 - 4.00 buvo įsivaizduojamas griebimas

            events = dict(left_hand=2, right_hand=1)
        elif self.paradigm == "ERP":
            interval = [
                0.0,
                1.0,
            ]  # [-0.2, 0.8] is the interval used in paper for online prediction
            events = dict(Target=1, NonTarget=2)
        elif self.paradigm == "SSVEP":
            interval = [0.0, 4.0]
            events = {
                "12.0": 1,
                "8.57": 2,
                "6.67": 3,
                "5.45": 4,
            }  
            # dict(up=1, left=2, right=3, down=4)
        else:
            raise ValueError('unknown paradigm "{}"'.format(paradigm))

        super().__init__(
            subjects=self.subjects,
            sessions_per_subject = 2,
            events=events,
            code="Lee2019 #20",
            interval=[],
            doi="",
            additional_info="paradigms = 'MI','ERP','SSVEP'"
            )
        
        self.log = setup_logger('Lee2019')
        self.code_suffix = self.paradigm
        self.train_run = train_run
        self.test_run = paradigm == "p300" if test_run is None else test_run
        self.resting_state = resting_state
        self.event_id = events

    def _translate_class(self, c):
        if self.paradigm == "MI":
            dictionary = dict(
                left_hand=["left"],
                right_hand=["right"],
            )
        elif self.paradigm == "ERP":
            dictionary = dict(
                Target=["target"],
                NonTarget=["nontarget"],
            )
        elif self.paradigm == "SSVP":
            dictionary = {
                "12.0": ["up"],
                "8.57": ["left"],
                "6.67": ["right"],
                "5.45": ["down"],
            }
        for k, v in dictionary.items():
            if c.lower() in v:
                return k
        raise ValueError('unknown class "{}" for "{}" paradigm'.format(c, self.paradigm))

    def _check_mapping(self, file_mapping):
        def raise_error():
            raise ValueError(
                "file_mapping ({}) different than events ({})".format(
                    file_mapping, self.event_id
                )
            )

        if len(file_mapping) != len(self.event_id):
            raise_error()
        for c, v in file_mapping.items():
            v2 = self.event_id.get(self._translate_class(c), None)
            if v != v2 or v2 is None:
                raise_error()

    _scalings = dict(eeg=1e-6, emg=1e-6, stim=1)  # to load the signal in Volts

    def _make_raw_array(self, signal, ch_names, ch_type, sfreq, verbose=False):
        ch_names = [np.squeeze(c).item() for c in np.ravel(ch_names)]
        if len(ch_names) != signal.shape[1]:
            raise ValueError
        info = create_info(
            ch_names=ch_names, ch_types=[ch_type] * len(ch_names), sfreq=sfreq
        )
        factor = self._scalings.get(ch_type)
        raw = RawArray(data=signal.transpose(1, 0) * factor, info=info, verbose=verbose)
        return raw

    def _get_single_run(self, data):
        sfreq = data["fs"].item()
        file_mapping = {c.item(): int(v.item()) for v, c in data["class"]}
        self._check_mapping(file_mapping)

        # Create RawArray
        raw = self._make_raw_array(data["x"], data["chan"], "eeg", sfreq)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Create EMG channels
        emg_raw = self._make_raw_array(data["EMG"], data["EMG_index"], "emg", sfreq)

        # Create stim chan
        event_times_in_samples = data["t"].squeeze()
        event_id = data["y_dec"].squeeze()
        stim_chan = np.zeros(len(raw))
        for i_sample, id_class in zip(event_times_in_samples, event_id):
            stim_chan[i_sample] += id_class
        stim_raw = self._make_raw_array(
            stim_chan[:, None], ["STI 014"], "stim", sfreq, verbose="WARNING"
        )

        # Add EMG and stim channels
        raw = raw.add_channels([emg_raw, stim_raw])
        if self.paradigm == "MI":
            
            events = mne.find_events(raw, stim_channel='STI 014')
            # Convert the events to annotations
            onsets = events[:, 0] / raw.info['sfreq']  # Convert from sample to seconds
            durations = np.full(len(events), 4.0)  # Set duration of all events to 4 seconds
            descriptions = [str(event_id) for event_id in events[:, 2]]  # Event descriptions as string

            # Create annotations from events
            annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

            # Add annotations to raw data
            raw.set_annotations(annotations)
            self.log.info("Added anotations to raw")

        return raw

    def _get_single_rest_run(self, data, prefix):
        sfreq = data["fs"].item()
        raw = self._make_raw_array(
            data["{}_rest".format(prefix)], data["chan"], "eeg", sfreq
        )
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subejct."""

        sessions = {}
        file_path_list = glob.glob(os.path.join(self.path, '*subj{:02}_EEG_{}.mat').format(subject,self.paradigm))


        for session in self.sessions:
            if self.train_run or self.test_run:
                mat = loadmat(file_path_list[self.sessions.index(session)])
                self.log.info(F"Loaded mat for {subject} subject, {session} session")
                
            session_name = str(session - 1)
            sessions[session_name] = {}
            if self.train_run:
                sessions[session_name]["1train"] = self._get_single_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0]
                )
            if self.test_run:
                sessions[session_name]["4test"] = self._get_single_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0]
                )
            if self.resting_state:
                prefix = "pre"
                sessions[session_name][f"3{prefix}TestRest"] = self._get_single_rest_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0], prefix
                )
                sessions[session_name][f"0{prefix}TrainRest"] = self._get_single_rest_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0], prefix
                )
                prefix = "post"
                sessions[session_name][f"5{prefix}TestRest"] = self._get_single_rest_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0], prefix
                )
                sessions[session_name][f"2{prefix}TrainRest"] = self._get_single_rest_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0], prefix
                )

        return sessions

    # def data_path(self, subject, path):

    def dowload_paradigm_data(self):
        from helper_functions import download_file

        for url in aditional_url:
            download_file(url,self.path,self.log)
        #dowload files
        for paradigm in [self.paradigm, 'Artifact']:
            for subject in self.subjects:
                for session in self.sessions:
                    url = "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_{3}.mat".format(
                        Lee2019_URL, session, subject, paradigm
                    )
                    download_file(url,self.path,self.log)

    def initialize(self):
        from helper_functions import save_raw_to_fif, find_unprocesed_paths
        self.dowload_paradigm_data()
        
        # Initialize the file processing counter
        files_processed_no = 0

        # Find unprocessed paths
        _, info = find_unprocesed_paths(self.path, marks=f'EEG_{self.paradigm}.mat', info=True)
        total_paths_no = info['Nr_paths']
        data_destination_dir = os.path.join(os.getcwd(), 'data', 'raw_fif', '20', self.paradigm)
        

        for subject_no in self.subjects:
            try:
                subject_raws = self._get_single_subject_data(subject_no)
                raw_1 = subject_raws['0']['1train']
                raw_2 = subject_raws['1']['1train']
                raw_1_name = "s{:02}.01".format(subject_no)
                raw_2_name = "s{:02}.02".format(subject_no)
            except Exception as e:
                print(f"An error occurred while getting data for subject {subject_no}: {e}")

            try:
                # Load the raw data and save it to FIF format
                save_raw_to_fif(raw_1, dataset_no=20, file_name=raw_1_name, optional_dir=data_destination_dir, logger=self.log)
                save_raw_to_fif(raw_2, dataset_no=20, file_name=raw_2_name, optional_dir=data_destination_dir, logger=self.log)
                self.log.info(f"Succesfuly saved {raw_1_name} and {raw_2_name} as .fif to {data_destination_dir}")
                
                files_processed_no += 2  # Increment counter only after successful processing
            except Exception as e:
                self.log.error(f"Failed to process files {raw_1_name},{raw_2_name} Error: {e}")

        # Check if all files were processed
        if files_processed_no == total_paths_no:
            self.log.info("All files have been processed.")
        else:
            self.log.error(f"Some files were not processed. Files processed: {files_processed_no}, Total files: {total_paths_no}")
            
    def load_one_raw_fif(self, subject, paradigm, run):
        '''Read and return the raw data for a single subject'''
        
        subject_id = 's{:02}.{:02}_raw.fif'.format(subject, run)
        path = os.path.join(os.getcwd(), 'data','raw_fif', '20', paradigm, subject_id)

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
            paradigms = "MI"
            # paradigms = ["MI","Artefact","ERP","SSVEP"]
            
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
                    subject_paths = glob.glob(os.path.join(os.getcwd(), 'data','raw_fif', '20', paradigm, 's{:02}.*_raw.fif').format(subject))
                    run_no = len(subject_paths)
                    
                    if run_no in {1, 2}:  # Check if it's within the expected range
                        current_runs = list(range(1, run_no + 1))
                    else:
                        self.log.error(f"Found {len(subject_paths)} files for subject {subject}. Expected 1 or 2 files.")
                        continue  # Skip this subject if the number of runs is unexpected
                data[paradigm][subject] = {}
                for run in [current_runs]:
                    raw = self.load_one_raw_fif(subject, run=run, paradigm=paradigm)
                    data[paradigm][subject][run] = raw
        return data