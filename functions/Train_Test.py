import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_and_print(data):
    gss = GroupShuffleSplit(test_size=0.3, n_splits=2, random_state=7)
    healthcodes = data['healthcode']
    y = data['y']

    train_ind, test_ind = next(gss.split(data, y, groups=healthcodes))

    train_data = data.iloc[train_ind]
    test_data = data.iloc[test_ind]

    for dataset, label in [(train_data, "Train"), (test_data, "Test")]:
        for y_val, y_label in [(0, "HC"), (1, "PD")]:
            data_y = dataset[dataset.y == y_val]

            for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
                data_yg = data_y[data_y.gender == gender_val]
                sample_size = data_yg.shape[0]
                subject_size = data_yg['healthcode'].nunique()
                print(
                    f"{label} data {y_label} {gender_label} - Sample Size: {sample_size}, Subject Size: {subject_size}")
            print('\n')
    return train_data, test_data


def split_and_print2(data):
    gss = GroupShuffleSplit(test_size=0.3, n_splits=2, random_state=7)
    healthcodes = data['healthcode']
    y = data['y']

    train_ind, test_ind = next(gss.split(data, y, groups=healthcodes))

    train_data = data.iloc[train_ind]
    test_data = data.iloc[test_ind]

    results = []

    for dataset, label in [(train_data, "Train"), (test_data, "Test")]:
        for y_val, y_label in [(0, "HC"), (1, "PD")]:
            data_y = dataset[dataset.y == y_val]

            for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
                data_yg = data_y[data_y.gender == gender_val]
                sample_size = data_yg.shape[0]
                subject_size = data_yg['healthcode'].nunique()
                results.append({
                    'Set': label,
                    'Condition': y_label,
                    'Gender': gender_label,
                    'Sample Size': sample_size,
                    'Subject Size': subject_size
                })

    # Create DataFrame and print it
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return train_data, test_data
