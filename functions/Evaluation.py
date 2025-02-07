from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report, make_scorer)
import pandas as pd
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def evaluate_model(model, data_test):
    x_test = data_test.drop(['age', 'healthcode', 'y', 'age_range', 'gender'], axis=1)
    y_test = data_test['y']

    # Predict on test set
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    # Print results for each age range
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Specificity: {specificity:.4f}')

    print('\nConfusion Matrix:')
    print(pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)


def cv_scores(x, y, heaqlthcode, model):
    specificity = make_scorer(specificity_score)
    # Define the scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'specificity': specificity
    }
    cv = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # Perform cross-validation
    results = cross_validate(model, x, y, groups=heaqlthcode, cv=cv, scoring=scoring, return_train_score=False)

    # Print the results
    for metric in scoring.keys():
        print(f"{metric}: {np.mean(results['test_' + metric]):.4f} (+/- {np.std(results['test_' + metric]):.4f})")
