import pandas as pd


def rec(data, lw, up):
    num = up - lw
    df_rec = data.groupby('healthcode').size().reset_index(name="num_recordings")
    data = pd.merge(data, df_rec, on='healthcode')
    df_rec_merge_less = data[(data.num_recordings <= up) & (data.num_recordings > lw)]
    df_rec_merge_more = data[data.num_recordings > up]
    df_rec_merge_more = df_rec_merge_more.groupby("healthcode").sample(n=num, random_state=1)
    data = pd.concat([df_rec_merge_less, df_rec_merge_more])
    data = data.drop(['num_recordings'], axis=1)
    data = data[[c for c in data if c not in ['age', 'gender', 'healthcode', 'y']]
                + ['age', 'gender', 'healthcode', 'y']]
    print("Filtered data size : {}".format(data.shape))
    return data




