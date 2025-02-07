# Parkinson's Disease Detection

## Interpretable Parkinson’s Disease Detection Using Group-Wise Scaling

This repository contains code and resources for detecting Parkinson's Disease using voice features extracted from publicly available datasets. The code is organized into multiple notebooks and scripts that handle data downloading, preprocessing, feature extraction, anomaly detection, resampling, scaling, feature selection, model training, evaluation, Interpretation.

## Repository Structure

```
├── Feature_Extractor.ipynb       # Main feature extraction notebook using OpenSmile eGeMAPS, emobse, ComParE
├── README.md                     # Project documentation
├── functions/                     # Utility scripts and helper functions
│   ├── Anomaly.py                 # Anomaly detection functions
│   ├── Evaluation.py               # Model evaluation functions
│   ├── Filter_record.py            # Data filtering functions (keep min of 1 and max 30 recordings per subject)
│   ├── Group_age_gender.py         # Functions for age and gender grouping
│   ├── GWS.py                      # Feature scaling functions using group-wise scaling
│   ├── __init__.py                 # Package initializer
│   ├── Print_data.py               # Data printing utilities
│   ├── read_data.py                # Data reading functions
│   ├── Resample.py                 # Data resampling functions
│   ├── RFECV_Func.py               # Feature selection using RFECV
│   ├── SHap_Func.py                # SHAP-based feature importance analysis
│   ├── Train_Test.py               # Train-test splitting functions
│
├── mPower/                         # mPower dataset processing
│   ├── DownloadmPower.ipynb        # Download mPower dataset
│   ├── mPowe_eGeMAPS.ipynb         # Parkinson's detection using eGeMAPS features mPower dataset
│   ├── mPower_ComParE.ipynb        # Parkinson's detection using ComParE features mPower dataset
│   ├── mPower_emobase.ipynb        # Parkinson's detection using emobase features mPower dataset
│   ├── mPower_metadata.ipynb       # Extract metadata of mPower data
│   ├── Normalized_and_16k_resample.ipynb # Normalize and resample audio
│   ├── Plot_ROC_mPower.ipynb       # Plot ROC curves for mPower dataset
│   ├── Rename_mpower_files.ipynb   # Rename files for consistency
│
├── Prior/                          # Prior dataset processing
│   ├── Prior_ComParE.ipynb         # Parkinson's detection using ComParE features Prior dataset
│   ├── Prior_eGeMAPS.ipynb         # Parkinson's detection using eGeMAPS features Prior dataset
│   ├── Prior_emobase.ipynb         # Parkinson's detection using emobase features Prior dataset
│
├── Vaiciukynas/                    # Vaiciukynas dataset processing
│   ├── Vaiciukynas_emobase_notgroupcv.ipynb   # Parkinson's detection using emobase features with the method in [19] (info leakage present)
│   ├── Vaiciukynas_emobase.ipynb   # Parkinson's detection using emobase features with the method in [19]
│   ├── Vaiciukynas_ktu.ipynb       # Parkinson's detection using ktu features
```

## Datasets

This project utilizes three publicly available datasets:

1. **mPower Dataset** (Mobile Parkinson Disease Study)  
   - Access: [mPower Data on Synapse](https://dhealth.synapse.org/Explore/Collections/DetailsPage?study=mPower%20Mobile%20Parkinson%20Disease%20Study#DataAccess)

2. **Prior Dataset** (Voice samples for patients with Parkinson's Disease and healthy controls)  
   - Access: [Prior Dataset on Figshare](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127)

3. **Vaiciukynas Dataset** (Voice-based Parkinson's Disease detection study)  
   - Access: [Vaiciukynas Dataset in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5628839/)

## Usage

### mPower

1. **Download and rename datasets**  
   - Register for a Synapse account and follow the steps on their website.
   - Download dataset using `DownloadmPower.ipynb` in batches to avoid memory issues. The mPower voice data contains 65000 recordings which is 87.2 GB.
   - Rename files using `Rename_mpower_files.ipynb` for consistency.

2. **Extract features**  
   - Use `Feature_Extractor.ipynb` to extract vocal features (eGeMAPS, emobase, ComParE) using OpenSmile.

3. **Normalize and resample to 16k**  
   - Use `Normalize_and_16k_resample.ipynb` to scale all the voice recordings by removing the mean and having a unit standard deviation.

4. **Perform Parkinson's Detection**  
   - Run `mPower_eGeMAPS.ipynb`, `mPower_emobase.ipynb`, `mPower_ComParE.ipynb` to train models.
   - The code first loads and preprocesses the dataset, including filtering out frequent recordings, splitting data into age groups (Young, Mid, and Old), and performing anomaly detection to clean the data. It then resamples the data to handle class imbalances and applies feature scaling using GWS (Group-Wise Scaling). An XGBoost classifier is trained on the scaled data, followed by feature selection using Recursive Feature Elimination with Cross-Validation (RFECV). The model is fine-tuned using GridSearchCV, and its performance is evaluated through accuracy, ROC curve analysis, and SHAP values for feature importance and interpretation.

### Prior

1. **Extract features**  
   - Use `Feature_Extractor.ipynb` for eGeMAPS, emobase, and ComParE feature extraction.

2. **Perform Parkinson's Detection**  
   - Use `Prior_eGeMAPS.ipynb`, `Prior_emobase.ipynb`, `Prior_ComParE.ipynb` to train models.
   - Apply feature scaling (GWS), train an XGBoost classifier, and evaluate results.

### Vaiciukynas

1. **Download datasets**  
   - Download zip file of data, in the form of extracted audio features from voice and speech recordings.

2. **Perform Parkinson's Detection**  
   - Use `Vaiciukynas_emobase.ipynb`, `Vaiciukynas_emobase_notgroupcv.ipynb`, and `Vaiciukynas_ktu.ipynb` to train models.
   - Apply feature scaling (GWS), train an XGBoost classifier, and evaluate results.

## Citation

If you use this repository in your research, please cite our paper:

```
@article{Momeni2025,
  author = {Momeni, Niloofar and Whitling, Susanna and Jakobsson, Andreas},
  title = {Interpretable Parkinson’s Disease Detection Using Group-Wise Scaling},
  journal = {IEEE Access},
  year = {2025},
  doi = {Your DOI}
}
```


