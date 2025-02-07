
age1 = 35
age2 = 55
import pandas as pd


def age3(data, age1, age2):
    age_ranges = [
        (data.age < age1, 0),
        ((data.age >= age1) & (data.age < age2), 1),
        (data.age >= age2, 2)
    ]

    results = []
    table = []

    for condition, age_range in age_ranges:
        data_subset = data[((data.y == 0) | (data.y == 1)) & condition]
        data_subset['age_range'] = age_range

        for y_val, y_label in [(0, "HC"), (1, "PD")]:
            data_y = data_subset[data_subset.y == y_val]

            for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
                data_yg = data_y[data_y.gender == gender_val]
                sample_size = data_yg.shape[0]
                subject_size = data_yg['healthcode'].nunique()
                table.append([y_label, gender_label, age_range, sample_size, subject_size])

        results.append(data_subset)

    df_table = pd.DataFrame(table, columns=["Group", "Gender", "Age Range", "Sample Size", "Subject Size"])

    # Display full DataFrame without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df_table)

    data_young, data_mid, data_old = results
    return data_young, data_mid, data_old


def age2(data, age):
    age_ranges = [
        (data.age < age, 1),
        (data.age >= age, 2)
    ]

    results = []
    table = []

    for condition, age_range in age_ranges:
        data_subset = data[((data.y == 0) | (data.y == 1)) & condition]
        data_subset['age_range'] = age_range

        for y_val, y_label in [(0, "HC"), (1, "PD")]:
            data_y = data_subset[data_subset.y == y_val]

            for gender_val, gender_label in [(0, "Female"), (1, "Male")]:
                data_yg = data_y[data_y.gender == gender_val]
                sample_size = data_yg.shape[0]
                subject_size = data_yg['healthcode'].nunique()
                table.append([y_label, gender_label, age_range, sample_size, subject_size])

        results.append(data_subset)

    # print(tabulate(table, headers=["Group", "Gender", "Age Range", "Sample Size", "Subject Size"], tablefmt="grid"))
    # Convert table to DataFrame for better display control
    df_table = pd.DataFrame(table, columns=["Group", "Gender", "Age Range", "Sample Size", "Subject Size"])

    # Display full DataFrame without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df_table)

    data_mid, data_old = results
    return data_mid, data_old
