# MI_BCI

**Undergraduate Neurobiophysics Thesis**  
**Vilnius University Life Sciences Center**  
**Program**: Neurobiophysics study program
**Author / Autorius**: Rokas Bertašius  
**Supervisor / Darbo vadovas**: Dr. Aleksandras Voicikas  

**Thesis Title**  
- **EN**: Evaluation of the impact of EEG data quality and processing strategies on motor imagery classification accuracy.  
- **LT**: EEG duomenų kokybės ir apdorojimo strategijų poveikio motorinės vaizduotės klasifikavimo tikslumui įvertinimas  

---

## Table of Contents

1. [Project Overview / Projekto apžvalga](#project-overview--projekto-apžvalga)  
2. [Data / Duomenys](#data--duomenys)  
3. [Repository Structure / Repozitorijos struktūra](#repository-structure--repozitorijos-struktūra)  
4. [Environment / Aplinka](#environment--aplinka)  
5. [Installation / Diegimas](#installation--diegimas)  
6. [Usage / Naudojimas](#usage--naudojimas)  
7. [Preprocessing / Išankstinis apdorojimas](#preprocessing--išankstinis-apdorojimas)  
8. [Feature Extraction / Bruožų išgavimas](#feature-extraction--bruozu-isgavimas)  
9. [Classification & Evaluation / Klasifikacija ir vertinimas](#classification--evaluation--klasifikacija-ir-vertinimas)  
10. [Statistical Analysis / Statistinė analizė](#statistical-analysis--statistinė-analizė)  
11. [Results / Rezultatai](#results--rezultatai)  
12. [Visualization / Vizualizacija](#visualization--vizualizacija)  
13. [Contributing / Dalyvavimas](#contributing--dalyvavimas)  
14. [License / Licencija](#license--licencija)  
15. [Contact / Kontaktai](#contact--kontaktai)  
16. [Template Notes / Šablono pastabos](#template-notes--šablono-pastabos)  

---

## Project Overview / Projekto apžvalga

**EN**  
This repository contains all code, notebooks and results for my Bachelor’s thesis. We investigate how different EEG cleaning strategies and feature‐extraction pipelines affect motor imagery (MI) classification accuracy and cross‐session transfer, using the free‐access Lee et al. (2019) MI EEG dataset.

**LT**  
Šioje repositorijoje rasite visą kodą reikiamą atkartoti bakalauro baigiamajam darbui. Tiriame, kaip skirtingos EEG valymo strategijos ir požymių išgavimas veikia motorinės vaizduotės klasifikavimo tikslumą bei modelio pernašą tarp sesijų, naudodami viešos prieigos Lee ir kt. (2019) duomenų rinkinį.

---

## Data / Duomenys

- **Dataset**: Lee et al. (2019) free‐access motor imagery EEG  
- **Format**: raw MATLAB `.mat` → converted to MNE‐compatible `.fif`  
- **Recording**: 64 EEG channels, two sessions per subject (recorded on separate days), X trials per class (fill in)  
- **Classes**: left hand, right hand (specify if more)  
- **Sampling rate**: 1000 Hz (fill in)  

Place all raw `.mat` files in `data/raw/`; the preprocessing script will convert and save cleaned `.fif` files to `data/processed/`.

---

## Repository Structure / Repozitorijos struktūra

```plain
MI_BCI/
├── data/
│   ├── raw/               # original .mat files
│   └── processed/         # band-pass filtered & artifact-cleaned .fif
├── code/                  # all Python scripts
│   ├── preprocessing.py   # band-pass/notch + wavelet artifact removal + epoch rejection
│   ├── features_csp.py
│   ├── features_fbcsp.py
│   ├── features_cssp.py
│   ├── features_ts.py
│   ├── features_logvar.py
│   ├── classify.py        # train & test classifiers
│   └── utils.py           # helper functions
├── notebooks/             # exploratory analysis & figure generation
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_classification.ipynb
│   └── 04_stats_analysis.ipynb
├── results/
│   ├── figures/           # .png/.svg plots
│   ├── tables/            # .csv summaries & stats outputs
│   └── models/            # saved model `.pkl` files
├── requirements.txt       # pip-installable dependencies
└── README.md              # this file
