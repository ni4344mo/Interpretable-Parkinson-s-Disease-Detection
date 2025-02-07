from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupShuffleSplit
from matplotlib import pyplot as plt


min_features_to_select = 1  # Minimum number of features to consider
# cv = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=0)


def rfecv_sel(x, y, healthcode, model, dataname):
    rfecv = RFECV(
        estimator=model,
        step=2,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(x, y, groups=healthcode)
    print(f"Optimal number of features: {rfecv.n_features_}")
    mask = rfecv.get_support()
    features = x.columns
    best_features = features[mask]

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure(figsize=(7, 3))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean validation accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"] * 100,
        yerr=rfecv.cv_results_["std_test_score"] * 100,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    filename = "RFECV" + dataname + ".png"
    filename2 = "RFECV" + dataname + ".pdf"
    # plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.savefig(filename)
    plt.savefig(filename2)

    plt.show()

    return best_features


def rfecv_sel7(x, y, healthcode, model, dataname):
    rfecv = RFECV(
        estimator=model,
        step=2,
        cv= GroupShuffleSplit(n_splits=7, test_size=0.2, random_state=0),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(x, y, groups=healthcode)
    print(f"Optimal number of features: {rfecv.n_features_}")
    mask = rfecv.get_support()
    features = x.columns
    best_features = features[mask]

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure(figsize=(7, 3))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean validation accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"] * 100,
        yerr=rfecv.cv_results_["std_test_score"] * 100,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    filename = "RFECV" + dataname + ".png"
    filename2 = "RFECV" + dataname + ".pdf"
    # plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.savefig(filename)
    plt.savefig(filename2)

    plt.show()

    return best_features