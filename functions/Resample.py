from sklearn.utils import resample
import pandas as pd
from tabulate import tabulate  # Importing tabulate for table formatting

def resample_subgroups(data):
    # Group by condition (HC or PD) and gender (Female or Male)
    grouped = data.groupby(['y', 'gender'])

    # Calculate sample sizes for each subgroup
    subgroup_sizes = grouped.size().reset_index(name='Sample Size')

    # Sort by sample size to find the smallest and second-smallest subgroups
    sorted_subgroups = subgroup_sizes.sort_values(by='Sample Size')

    # Extract the sizes
    min_size = sorted_subgroups.iloc[0]['Sample Size']
    second_min_size = sorted_subgroups.iloc[1]['Sample Size']
    second_min_size = int(second_min_size)

    print(f"Minimum size: {min_size}, Second minimum size: {second_min_size}")

    # Initialize list to store resampled data frames
    resampled_data = []
    table_rows = []
    # Resample each subgroup to match the second-smallest subgroup size
    for idx, row in subgroup_sizes.iterrows():
        condition = row['y']
        gender = row['gender']
        group_data = grouped.get_group((condition, gender))

        current_size = row['Sample Size']

        if current_size > second_min_size:
            # Downsample if current size is larger than second_min_size
            resampled_group = resample(group_data, replace=False, n_samples=second_min_size, random_state=42)
        else:
            # Upsample if current size is smaller than second_min_size
            resampled_group = resample(group_data, replace=True, n_samples=second_min_size, random_state=42)

        # Calculate resampled sample size and subject size
        resampled_sample_size = resampled_group.shape[0]
        resampled_subject_size = resampled_group['healthcode'].nunique()


        resampled_data.append(resampled_group)
        # Append to table rows
        table_rows.append([f"Condition {condition}", 'Female' if gender == 0 else 'Male',
                           resampled_sample_size, resampled_subject_size])

    # Print the table
    headers = ['Condition', 'Gender', 'Resampled Sample Size', 'Resampled Subject Size']
    print("\nResampled Data Summary:")
    print(tabulate(table_rows, headers=headers, tablefmt='grid'))

    # Concatenate all resampled subgroups into a single DataFrame
    resampled_data_concat = pd.concat(resampled_data, ignore_index=True)

    return resampled_data_concat



def resample_subgroups3(data):
    # Group by condition (HC or PD) and gender (Female or Male)
    grouped = data.groupby(['y', 'gender'])

    # Calculate sample sizes for each subgroup
    subgroup_sizes = grouped.size().reset_index(name='Sample Size')

    # Sort by sample size to find the smallest and second-smallest subgroups
    sorted_subgroups = subgroup_sizes.sort_values(by='Sample Size')

    # Extract the sizes
    min_size = sorted_subgroups.iloc[0]['Sample Size']
    second_min_size = sorted_subgroups.iloc[2]['Sample Size']
    second_min_size = int(second_min_size)

    print(f"Minimum size: {min_size}, Second minimum size: {second_min_size}")

    # Initialize list to store resampled data frames
    resampled_data = []
    table_rows = []
    # Resample each subgroup to match the second-smallest subgroup size
    for idx, row in subgroup_sizes.iterrows():
        condition = row['y']
        gender = row['gender']
        group_data = grouped.get_group((condition, gender))

        current_size = row['Sample Size']

        if current_size > second_min_size:
            # Downsample if current size is larger than second_min_size
            resampled_group = resample(group_data, replace=False, n_samples=second_min_size, random_state=42)
        else:
            # Upsample if current size is smaller than second_min_size
            resampled_group = resample(group_data, replace=True, n_samples=second_min_size, random_state=42)

        # Calculate resampled sample size and subject size
        resampled_sample_size = resampled_group.shape[0]
        resampled_subject_size = resampled_group['healthcode'].nunique()


        resampled_data.append(resampled_group)
        # Append to table rows
        table_rows.append([f"Condition {condition}", 'Female' if gender == 0 else 'Male',
                           resampled_sample_size, resampled_subject_size])

    # Print the table
    headers = ['Condition', 'Gender', 'Resampled Sample Size', 'Resampled Subject Size']
    print("\nResampled Data Summary:")
    print(tabulate(table_rows, headers=headers, tablefmt='grid'))

    # Concatenate all resampled subgroups into a single DataFrame
    resampled_data_concat = pd.concat(resampled_data, ignore_index=True)

    return resampled_data_concat


def resample_subgroups2(data):
    # Group by condition (HC or PD) and gender (Female or Male)
    grouped = data.groupby(['y'])

    # Calculate sample sizes for each subgroup
    subgroup_sizes = grouped.size().reset_index(name='Sample Size')

    # Sort by sample size to find the smallest and second-smallest subgroups
    sorted_subgroups = subgroup_sizes.sort_values(by='Sample Size')

    # Extract the sizes
    min_size = sorted_subgroups.iloc[0]['Sample Size']
    second_min_size = sorted_subgroups.iloc[0]['Sample Size']
    second_min_size = int(second_min_size)

    print(f"Minimum size: {min_size}, Second minimum size: {second_min_size}")

    # Initialize list to store resampled data frames
    resampled_data = []
    table_rows = []
    # Resample each subgroup to match the second-smallest subgroup size
    for idx, row in subgroup_sizes.iterrows():
        condition = row['y']
        group_data = grouped.get_group((condition))

        current_size = row['Sample Size']

        if current_size > second_min_size:
            # Downsample if current size is larger than second_min_size
            resampled_group = resample(group_data, replace=False, n_samples=second_min_size, random_state=42)
        else:
            # Upsample if current size is smaller than second_min_size
            resampled_group = resample(group_data, replace=True, n_samples=second_min_size, random_state=42)

        # Calculate resampled sample size and subject size
        resampled_sample_size = resampled_group.shape[0]
        resampled_subject_size = resampled_group['healthcode'].nunique()


        resampled_data.append(resampled_group)
        # Append to table rows

    # Concatenate all resampled subgroups into a single DataFrame
    resampled_data_concat = pd.concat(resampled_data, ignore_index=True)

    return resampled_data_concat