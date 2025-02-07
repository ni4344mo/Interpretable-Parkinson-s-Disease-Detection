import pandas as np

def printdata(data):
    data_hc = data[data.y == 0]
    data_pd = data[data.y == 1]
    data_hf = data[(data.y == 0) & (data.gender == 0)]
    data_hm = data[(data.y == 0) & (data.gender == 1)]
    data_pf = data[(data.y == 1) & (data.gender == 0)]
    data_pm = data[(data.y == 1) & (data.gender == 1)]
    print("Total Size: {}, Subject Size: {}".format(data.shape[0], data['healthcode'].nunique()))
    print("HC - Sample Size: {}, Subject Size: {}".format(data_hc.shape[0], data_hc['healthcode'].nunique()))
    print("PD - Sample Size: {}, Subject Size: {}".format(data_pd.shape[0], data_pd['healthcode'].nunique()))
    # print('\n')
    print("HC Female - Sample Size: {}, Subject Size: {}".format(data_hf.shape[0], data_hf['healthcode'].nunique()))
    print("HC Male - Sample Size: {}, Subject Size: {}".format(data_hm.shape[0], data_hm['healthcode'].nunique()))
    print("PD Female - Sample Size: {}, Subject Size: {}".format(data_pf.shape[0], data_pf['healthcode'].nunique()))
    print("PD Male - Sample Size: {}, Subject Size: {}".format(data_pm.shape[0], data_pm['healthcode'].nunique()))
