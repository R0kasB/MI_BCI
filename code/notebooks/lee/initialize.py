import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')
from helper_functions import setup_logger
from datasets import Lee2019

log = setup_logger("Lee_preprocess")

dataset = Lee2019()

dataset.initialize()
