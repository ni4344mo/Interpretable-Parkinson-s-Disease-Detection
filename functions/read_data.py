import pandas as pd
import numpy as np


def read_mPower16knorm_eGeMAPS():
    meta_mPow = pd.read_csv('path_to_mPower_meta_data/mPower/Data_meta_mpower.csv')
    dm = pd.read_csv('path_to_mPower_extracted_feature_data/mPower/mPower16knorm_eGeMAPS_index.csv')
    dm = dm.reset_index(level=0)
    dm['audio_audio.m4a'] = dm.file.str.slice(41, 48)
    dm = dm.drop(['end', 'start', 'file', 'index'], axis=1)
    dm['audio_audio.m4a'] = dm['audio_audio.m4a'].astype('int')

    mPow = pd.merge(dm, meta_mPow, on='audio_audio.m4a')
    mPow = mPow.rename(columns={"healthCode": "healthcode", "professional_diagnosis": "y"})

    mPow = mPow[mPow['gender'].notna()]
    mPow = mPow[mPow['age'].notna()]
    mPow = mPow[(mPow.gender == 0.0) | (mPow.gender == 1.0)]
    # mPow = mPow[mPow['medTimepoint'].notna()]
    # col = mPow.pop('medTimepoint')
    # mPow.insert(0, col.name, col)
    mPow = mPow.drop(
        ['diagnosis_year', 'medication_start_year', 'onset_year', 'surgery', 'deep_brain_stimulation', 'race',
         'audio_audio.m4a', 'recordId', 'years_smoking', 'last_smoked', 'packs_per_day', 'smoked'], axis=1)
    return mPow


def read_mPower16knorm_emobase():
    meta_mPow = pd.read_csv('path_to_mPower_meta_data/mPower/Data_meta_mpower.csv')
    dm = pd.read_csv('path_to_mPower_extracted_feature_data/mPower/mPower16knorm_emobase_240321_index.csv')
    dm = dm.reset_index(level=0)
    dm['audio_audio.m4a'] = dm.file.str.slice(41, 48)
    dm = dm.drop(['end', 'start', 'file', 'index'], axis=1)
    dm['audio_audio.m4a'] = dm['audio_audio.m4a'].astype('int')

    mPow = pd.merge(dm, meta_mPow, on='audio_audio.m4a')
    mPow = mPow.rename(columns={"healthCode": "healthcode", "professional_diagnosis": "y"})

    mPow = mPow[mPow['gender'].notna()]
    mPow = mPow[mPow['age'].notna()]
    mPow = mPow[(mPow.gender == 0) | (mPow.gender == 1)]
    # mPow = mPow[mPow['medTimepoint'].notna()]
    # col = mPow.pop('medTimepoint')
    # mPow.insert(0, col.name, col)
    mPow = mPow.drop(
        ['diagnosis_year', 'medication_start_year', 'onset_year', 'surgery', 'deep_brain_stimulation', 'race',
         'audio_audio.m4a', 'recordId', 'years_smoking', 'last_smoked', 'packs_per_day', 'smoked'], axis=1)
    return mPow


def read_mPower16knorm_ComParE():
    meta_mPow = pd.read_csv('path_to_mPower_meta_data/mPower/Data_meta_mpower.csv')
    dm = pd.read_csv('path_to_mPower_extracted_feature_data/mPower/mPower16knorm_ComParE_240321_index.csv')
    dm = dm.reset_index(level=0)
    dm['audio_audio.m4a'] = dm.file.str.slice(41, 48)
    dm = dm.drop(['end', 'start', 'file', 'index'], axis=1)
    dm['audio_audio.m4a'] = dm['audio_audio.m4a'].astype('int')

    mPow = pd.merge(dm, meta_mPow, on='audio_audio.m4a')
    mPow = mPow.rename(columns={"healthCode": "healthcode", "professional_diagnosis": "y"})

    mPow = mPow[mPow['gender'].notna()]
    mPow = mPow[mPow['age'].notna()]
    mPow = mPow[(mPow.gender == 0) | (mPow.gender == 1)]
    # mPow = mPow[mPow['medTimepoint'].notna()]
    # col = mPow.pop('medTimepoint')
    # mPow.insert(0, col.name, col)
    mPow = mPow.drop(
        ['diagnosis_year', 'medication_start_year', 'onset_year', 'surgery', 'deep_brain_stimulation', 'race',
         'audio_audio.m4a', 'recordId', 'years_smoking', 'last_smoked', 'packs_per_day', 'smoked'], axis=1)
    return mPow




def read_fred_eGeMAPS():
    # Load and clean the 'meta' data
    meta = pd.read_excel('/home/ni4344mo/data/PD data public/Fred/Demographics_age_sex.xlsx')
    meta = meta.rename(columns={'Sex': 'gender', 'Sample ID': 'healthcode',
                                'Age': 'age', 'Label': 'y'})
    meta['gender'] = meta['gender'].replace({'F': 0, 'M': 1})
    meta['y'] = meta['y'].replace({'HC': 0, 'PwPD': 1})

    # Load 'hc' and 'par' data, clean, and add target variable 'y'
    hc = pd.read_csv(
        '/home/ni4344mo/PycharmProjects/pythonProject2/Fred/HC_Fred_eGeMAPS_240916_index.csv').reset_index()
    par = pd.read_csv(
        '/home/ni4344mo/PycharmProjects/pythonProject2/Fred/PD_Fred_eGeMAPS_240916_index.csv').reset_index()

    for df in [hc, par]:
        df['healthcode'] = df['file'].apply(lambda x: x.split('/')[-1].split('.')[0])
        df.drop(columns=['end', 'start', 'file', 'index'], inplace=True)

    # hc['y'] = 0  # Healthy Control
    # par['y'] = 1  # Parkinson's Disease

    # Concatenate 'hc' and 'par' into a single DataFrame
    df = pd.concat([hc, par], ignore_index=True)

    # Merge 'df' with 'meta' on 'healthcode'
    data = pd.merge(df, meta, on='healthcode', how='left')

    # Remove rows where 'gender' or 'age' are NaN and exclude specific 'healthcode' values
    data = data.dropna(subset=['gender', 'age'])
    return data

def read_fred_emobase():
    # Load and clean the 'meta' data
    meta = pd.read_excel('/home/ni4344mo/data/PD data public/Fred/Demographics_age_sex.xlsx')
    meta = meta.rename(columns={'Sex': 'gender', 'Sample ID': 'healthcode',
                                'Age': 'age', 'Label': 'y'})
    meta['gender'] = meta['gender'].replace({'F': 0, 'M': 1})
    meta['y'] = meta['y'].replace({'HC': 0, 'PwPD': 1})

    # Load 'hc' and 'par' data, clean, and add target variable 'y'
    hc = pd.read_csv(
        '/home/ni4344mo/PycharmProjects/pythonProject2/Fred/HC_Fred_emobase_240916_index.csv').reset_index()
    par = pd.read_csv(
        '/home/ni4344mo/PycharmProjects/pythonProject2/Fred/PD_Fred_emobase_240916_index.csv').reset_index()

    for df in [hc, par]:
        df['healthcode'] = df['file'].apply(lambda x: x.split('/')[-1].split('.')[0])
        df.drop(columns=['end', 'start', 'file', 'index'], inplace=True)

    # hc['y'] = 0  # Healthy Control
    # par['y'] = 1  # Parkinson's Disease

    # Concatenate 'hc' and 'par' into a single DataFrame
    df = pd.concat([hc, par], ignore_index=True)

    # Merge 'df' with 'meta' on 'healthcode'
    data = pd.merge(df, meta, on='healthcode', how='left')

    # Remove rows where 'gender' or 'age' are NaN and exclude specific 'healthcode' values
    data = data.dropna(subset=['gender', 'age'])
    return data

def read_fred_ComParE():
    # Load and clean the 'meta' data
    meta = pd.read_excel('/home/ni4344mo/data/PD data public/Fred/Demographics_age_sex.xlsx')
    meta = meta.rename(columns={'Sex': 'gender', 'Sample ID': 'healthcode',
                                'Age': 'age', 'Label': 'y'})
    meta['gender'] = meta['gender'].replace({'F': 0, 'M': 1})
    meta['y'] = meta['y'].replace({'HC': 0, 'PwPD': 1})

    # Load 'hc' and 'par' data, clean, and add target variable 'y'
    hc = pd.read_csv(
        '/home/ni4344mo/PycharmProjects/pythonProject2/Fred/HC_Fred_ComParE_240916_index.csv').reset_index()
    par = pd.read_csv(
        '/home/ni4344mo/PycharmProjects/pythonProject2/Fred/PD_Fred_ComParE_240916_index.csv').reset_index()

    for df in [hc, par]:
        df['healthcode'] = df['file'].apply(lambda x: x.split('/')[-1].split('.')[0])
        df.drop(columns=['end', 'start', 'file', 'index'], inplace=True)

    # hc['y'] = 0  # Healthy Control
    # par['y'] = 1  # Parkinson's Disease

    # Concatenate 'hc' and 'par' into a single DataFrame
    df = pd.concat([hc, par], ignore_index=True)

    # Merge 'df' with 'meta' on 'healthcode'
    data = pd.merge(df, meta, on='healthcode', how='left')

    # Remove rows where 'gender' or 'age' are NaN and exclude specific 'healthcode' values
    data = data.dropna(subset=['gender', 'age'])
    return data
