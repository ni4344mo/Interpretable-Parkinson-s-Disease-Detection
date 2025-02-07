import pandas as pd
from sklearn.ensemble import IsolationForest


def clean_anomaly_data(data, th1, th2):
    # Separate hc and pd data
    pd_data = data[data.y == 1]
    hc_data = data[data.y == 0]

    # Drop unnecessary columns for hc
    xhc = hc_data.drop(['healthcode', 'y'], axis=1)
    xpd = pd_data.drop(['healthcode', 'y'], axis=1)

    # Initialize and fit the IsolationForest model for hc
    model_hc = IsolationForest(n_estimators=100, max_samples='auto', random_state=42)
    model_hc.fit(xhc.values)

    # Calculate anomaly scores for hc
    hc_data['anomaly_scores'] = model_hc.decision_function(xhc.values)

    # Remove hc data with anomaly scores lower than 0
    clean_hc = hc_data[hc_data.anomaly_scores >= th1].drop('anomaly_scores', axis=1)
    hc_removed = len(hc_data) - len(clean_hc)

    # Initialize and fit the IsolationForest model for pd
    model_pd = IsolationForest(n_estimators=100, max_samples='auto', random_state=42)
    model_pd.fit(xpd.values)

    # Calculate anomaly scores for pd
    pd_data['anomaly_scores'] = model_pd.decision_function(xpd.values)

    # Remove pd data with anomaly scores lower than 0
    clean_pd = pd_data[pd_data.anomaly_scores >= th2].drop('anomaly_scores', axis=1)
    pd_removed = len(pd_data) - len(clean_pd)

    # Concatenate the cleaned hc and pd data
    clean_data = pd.concat([clean_hc, clean_pd], ignore_index=True)
    total_removed = len(data) - len(clean_data)

    # Print how much data was removed
    print(f"Data removed from HC: {hc_removed} out of {len(hc_data)}")
    print(f"Data removed from PD: {pd_removed} out of {len(pd_data)}")
    print(f"Data removed total: {total_removed} out of {len(data)}")
    print(f"Total data: {len(data)}")
    print(f"Clean data: {len(clean_data)}")

    return clean_data  # returning hc_data and pd_data for plotting


# # Assuming mPow is your dataset
# clean_mpow, hc_data, pd_data = clean_anomaly_data(mPow)
#
# # Plotting the anomaly scores for hc and pd
# plt.figure(figsize=(14, 6))
#
# plt.subplot(1, 2, 1)
# sns.histplot(hc_data['anomaly_scores'], kde=True, color='blue')
# plt.title('Anomaly Scores for HC')
# plt.xlabel('Anomaly Score')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 2, 2)
# sns.histplot(pd_data['anomaly_scores'], kde=True, color='red')
# plt.title('Anomaly Scores for PD')
# plt.xlabel('Anomaly Score')
# plt.ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()
#
# # Display the first few rows of the cleaned data
# print(clean_mpow.head())

def anomaly_all(data, th):
    # Drop unnecessary columns for hc
    x = data.drop(['healthcode', 'y'], axis=1)

    # Initialize and fit the IsolationForest model for hc
    model = IsolationForest(n_estimators=100, max_samples='auto', random_state=42)
    model.fit(x.values)

    # Calculate anomaly scores for hc
    data['anomaly_scores'] = model.decision_function(x.values)

    # Remove hc data with anomaly scores lower than 0
    clean = data[data.anomaly_scores >= th].drop('anomaly_scores', axis=1)
    removed = len(data) - len(clean)

    # Print how much data was removed
    print(f"Data removed total: {removed} out of {len(data)}")
    print(f"Total data: {len(data)}")
    print(f"Clean data: {len(clean)}")

    return clean, model  # returning hc_data and pd_data for plotting


def anomaly_test(data, model, th):
    # Drop unnecessary columns for hc
    x = data.drop(['healthcode', 'y'], axis=1)

    # Calculate anomaly scores for hc
    data['anomaly_scores'] = model.decision_function(x.values)

    # Remove hc data with anomaly scores lower than 0
    clean = data[data.anomaly_scores >= th].drop('anomaly_scores', axis=1)
    removed = len(data) - len(clean)

    # Print how much data was removed
    print(f"Data removed total: {removed} out of {len(data)}")
    print(f"Total data: {len(data)}")
    print(f"Clean data: {len(clean)}")

    return clean
