# MI_BCI

**Undergraduate Neurobiophysics Thesis**  
**Vilnius University Life Sciences Center**  
**BSc program**: Neurobiophysics  
**Author / Autorius**: Rokas Bertašius  
**Supervisor / Darbo vadovas**: Dr. Aleksandras Voicikas  

**Thesis Title**  
- **EN**: Evaluation of the impact of EEG data quality and processing strategies on motor imagery classification accuracy.  
- **LT**: EEG duomenų kokybės ir apdorojimo strategijų poveikio motorinės vaizduotės klasifikavimo tikslumui įvertinimas   

**Feel free to ask any questions**: [rokasbertasius@gmail.com](mailto:rokasbertasius@gmail.com)  
---

## Table of Contents

1. [Project Overview / Projekto apžvalga](#project-overview--projekto-apžvalga)  
2. [Data / Duomenys](#data--duomenys)  
3. [Repository Structure / Repozitorijos struktūra](#repository-structure--repozitorijos-struktūra)  
4. [Setup / Paruošimas](#setup--paruošimas)  
5. [Usage / Naudojimas](#usage--naudojimas)   



---

## Project Overview / Projekto apžvalga

**EN**  
This repository contains all code, notebooks and results for my Bachelor’s thesis. We investigate how different EEG cleaning strategies and feature‐extraction pipelines affect motor imagery (MI) classification accuracy and cross‐session transfer, using the free‐access Lee et al. (2019) MI EEG dataset.

**LT**  
Šioje repositorijoje rasite visą kodą reikiamą atkartoti bakalauro baigiamajam darbui. Tiriame, kaip skirtingos EEG valymo strategijos ir požymių išgavimas veikia motorinės vaizduotės klasifikavimo tikslumą bei modelio pernašą tarp sesijų, naudodami viešos prieigos Lee ir kt. (2019) duomenų rinkinį.

---

## Data / Duomenys  
Data used in the analysis / Analizėje naudoti duomenys  

- **Dataset**: Lee et al. (2019) free‐access motor imagery EEG  
- **Format**: raw MATLAB `.mat` → converted to MNE‐compatible `.fif`  
- **Recording**: 64 EEG channels, two sessions per subject (recorded on separate days), 100 trials per class 
- **Classes**: left hand, right hand 
- **Sampling rate**: 1000 Hz

---

## Repository Structure / Repozitorijos struktūra

```plain
MI_BCI/
├── code
│      ├── datasets
│      │      └── *.py
│      ├── evaluation
│      │      └── *.py
│      ├── helper_functions
│      │      ├── eda
│      │      │      └── *.py
│      │      ├── data_processing
│      │      │      └── *.py
│      │      └── *.py
│      ├── notebooks
│      │       └── tutorials
│      │                └── *.ipynb
│      └── pipelines
│               └── *.py
├── plots
│      └── *.png
├── pixi.lock
├── pixi.toml
└── README.md              # this file
```
---

## Setup / Paruošimas

All dependencies and Python settings are declared in the [`pixi.toml`](pixi.toml) file.  
For more on Pixi, see [https://prefix.dev/](https://prefix.dev/).  

To recreate the exact environment:

```bash
# 1. Install the Pixi CLI (if you haven't already)
curl -fsSL https://pixi.sh/install.sh | sh   # macOS/Linux
# or PowerShell:
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"

# 2. Change into the directory containing pixi.toml
cd /path/to/MI_BCI

# 3. Install all dependencies from pixi.toml
pixi install

# 4. Activate the virtual environment
pixi shell
```
---

## Usage / Naudojimas
 
To run the tutorial notebooks and explore the workflows:

1. Open a terminal and change into the tutorials folder:  
   ```bash
   cd code/notebooks/tutorials

