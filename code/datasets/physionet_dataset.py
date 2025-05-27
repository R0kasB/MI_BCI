# This software includes code from the "moabb" project, licensed under the BSD 3-Clause License.
# 
# Copyright (c) 2017, authors of moabb
# All rights reserved.

import mne
import os
from datasets import BaseDataset
from helper_functions import setup_logger
import glob


class Physionet(BaseDataset):
    '''Class to load the Physionet dataset'''
    
    def __init__(self, subjects=None, path=None, run=None):
        self.subjects = subjects if subjects else list(range(1, 110))
        self.path = path if path else os.path.join(os.getcwd(), 'data','raw', '2')

        self.run = run if run else list(range(1, 15))
        self.run_info = {
            "task": "run",
            "Baseline, eyes open": 1,
            "Baseline, eyes closed": 2,
            "Motor execution: left vs right hand": [3, 7, 11],
            "Motor imagery: left vs right hand": [4, 8, 12],
            "Motor execution: hands vs feet": [5, 9, 13],
            "Motor imagery: hands vs feet": [6, 10, 14]
            }
        
        super().__init__(    
            subjects=self.subjects,
            sessions_per_subject = self.run, 
            events = dict(left_hand=2, right_hand=3, feet=5, hands=4, rest=1), #is moabb
            code = "Physionet #2",
            interval = [0, 3], #is moabb
            doi = "10.1109/TBME.2004.827072",
            additional_info = f"run_info: {self.run_info}"
        )
        self.log = setup_logger('Physionet')


    # def _download_all_data(self):
    #     '''Load data for all specified subjects'''
    #     [mne.datasets.eegbci.load_data(subject, list(range(1, 15)), path=self.path) for subject in range(1, 110)]
    #     self.log.info("Data downloaded successfully")

    def _load_one_raw(self, subject, runs):
        '''Load and return the raw data for a single subject'''
        run_dict = {}
        raw_files = mne.datasets.eegbci.load_data(subject, runs, path=self.path) #load_data dowloads data if not presant localy
        for f, run in zip(raw_files, runs):
            raw = mne.io.read_raw_edf(f, preload=True)
        
            raw.rename_channels(lambda x: x.strip("."))
            raw.rename_channels(lambda x: x.upper())
            # fmt: off
            renames = {
                "AFZ": "AFz", "PZ": "Pz", "FPZ": "Fpz", "FCZ": "FCz", "FP1": "Fp1", "CZ": "Cz",
                "OZ": "Oz", "POZ": "POz", "IZ": "Iz", "CPZ": "CPz", "FP2": "Fp2", "FZ": "Fz",
            }
            
            # fmt: on
            raw.rename_channels(renames)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
            run_dict[f'{run}'] = raw
        return run_dict

    def initialize(self):
        from helper_functions import save_raw_to_fif
        
        # self._download_all_data()

        files_processed_no = 0

        data_destination_dir = os.path.join(os.getcwd(), 'data', 'raw_fif', '2')

        def _subject_iteration(constructed_dir, runs, files_processed_no):
            for subject in self.subjects:
                run_dict = self._load_one_raw(subject, runs)
                for run in runs:
                    raw = run_dict[f'{run}']
                    save_raw_to_fif(raw, 2, f's{subject}.{run}',optional_dir=constructed_dir, logger = self.log)
                    files_processed_no +=1
            return files_processed_no
        
        def _task_iteration(specified_dir, specified_subdir, runs, files_processed_no):
            constructed_dir = os.path.join(data_destination_dir, specified_dir, specified_subdir)
            _subject_iteration(constructed_dir, runs, files_processed_no)
            return files_processed_no

        files_processed_no = _task_iteration('baseline', 'eyes_closed', runs=[2], files_processed_no=files_processed_no)
        files_processed_no = _task_iteration('baseline', 'eyes_open', runs=[1], files_processed_no=files_processed_no)

        files_processed_no = _task_iteration('motor_imagery', 'left_right', runs=[4, 8, 12], files_processed_no=files_processed_no)
        files_processed_no = _task_iteration('motor_imagery', 'hands_feet', runs=[6, 10, 14], files_processed_no=files_processed_no)

        files_processed_no = _task_iteration('motor_execution', 'left_right', runs=[3, 7, 11], files_processed_no=files_processed_no)
        files_processed_no = _task_iteration('motor_execution', 'hands_feet', runs=[5, 9, 13], files_processed_no=files_processed_no)

        # Check if all files were processed
        total_files = len(self.subjects) * len(self.run)
        if files_processed_no == total_files:
            self.log.info("All files have been processed.")
        else:
            self.log.error(f"Some files were not processed. Files processed: {files_processed_no}, Total files: {total_files}")
        
    def load_one_raw_fif(self, subject, paradigm,test):
        '''Read and return the raw data for a single subject'''
        try:
            path = glob.glob(os.path.join(os.getcwd(), 'data','raw_fif', '2', paradigm, test , f's{subject}.*_raw.fif'))[0]
        except IndexError as e:
            self.log.error(f"Failed to find raw data for subject {subject}: {e}")
            return None
        try:
            raw = mne.io.read_raw_fif(path, preload=True)
        except FileNotFoundError as e:
            self.log.error(f"Failed to load raw data for subject {subject}: {e}")
            raw = None  # You may choose to return None or handle it differently
        return raw


    def load_fif_data(self, subjects=None, paradigms=None, tests=None):
        if subjects is None:
            subjects = self.subjects
        if paradigms is None:
            paradigms = ['baseline', 'motor_execution', 'motor_imagery']
        test_condition = 0
        if isinstance(subjects, int):
            subjects = [subjects]
        if isinstance(paradigms, str):
            paradigms = [paradigms]
        if isinstance(tests, str):
            tests = [tests]
            test_condition = 1
            
        data = {}
        for paradigm in paradigms:
            data[paradigm] = {}
            if test_condition == 0:
                if paradigm == 'baseline':
                    tests = ['eyes_closed', 'eyes_open']
                elif paradigm == 'motor_execution':
                    tests = ['left_right', 'hands_feet']
                elif paradigm == 'motor_imagery':
                    tests = ['left_right', 'hands_feet']
            for test in tests:
                current_test = test
                data[paradigm][current_test] = {}
                for subject in subjects:
                    raw = self.load_one_raw_fif(subject=subject,paradigm=paradigm,test=test)
                    data[paradigm][current_test][subject] = raw
        return data