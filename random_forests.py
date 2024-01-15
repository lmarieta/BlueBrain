from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_curve, classification_report,
                             confusion_matrix, auc, precision_recall_curve)
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from statistics import mean
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from io import StringIO


def get_distinct_colors(num_colors):
    # Create a color map from the available colors
    colormap = plt.get_cmap("tab20")

    # Generate a list of distinct colors
    colors = [colormap(i) for i in range(num_colors)]

    return colors


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['peak_voltage', 'ISI_values', 'min_AHP_time', 'min_AHP_voltage', 'AP_begin_voltage',
                     'AP_begin_time', 'AP_amplitude', 'AP_width', 'AP_rise_time', 'AP_fall_time', 'time_to_AP_peak',
                     'AHP_duration', 'AHP_fall_tau', 'AHP_fall_A', 'AHP_rise_m', 'AHP_rise_c']
    threshold_detector = 'spikecount'
    protocols = ['APWaveform']
    data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                              threshold_detector=threshold_detector, protocols_to_plot=protocols)

    random_state = 0

    # Exclude IN cells and merge PC-L2 with PC-L5
    data = data[data['CellTypeGroup'] != 'IN']
    data['Group without cell type'] = data['Species'] + '_' + data['BrainArea']

    # Split your data into training and validation sets
    X = data[feature_names]

    y = data['Group without cell type']

    # Use SMOTE to oversample the minority class
    oversample = SMOTE(random_state=random_state)
    over_X, over_y = oversample.fit_resample(X, y)

    # Split your data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(over_X, over_y, test_size=0.3, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Initialize the hyperparameters
    n_estimator = 100  # TODO cross-validation and choose best trade-off between bias and variance or increase
    # TODO trees until out-of-bag error stabilizes or start to increase
    criterion = 'log_loss'  # log_loss because we are working with multi-class classification (as opposed to binary)
    max_depth = 10  # TODO cross-validation and minimize out-of-bag error
    min_samples_split = 2  # Increase to reduce overfitting
    min_samples_leaf = 1  # Increase to reduce overfitting
    min_weight_fraction_leaf = 0  # Increase in case of highly imbalanced dataset
    max_features = 'sqrt'  # Can be used to reduce the number of features
    max_leaf_nodes = None  # Set a value to reduce complexity
    min_impurity_decrease = 0.0  # Set a value to reduce complexity
    bootstrap = True  # Build trees with random sub-sample from dataset if True, otherwise use whole dataset
    oob_score = False  # If true estimate the performance of the model without the need for a separate validation set
    warm_start = False  # When set to True, reuse the solution of the previous call to fit and add more estimators
    class_weight = 'balanced'  # Trying balanced because of imbalanced dataset
    ccp_alpha = 0  # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost
    # complexity that is smaller than ccp_alpha will be chosen.
    max_samples = None  # None means each training set as the same size as original set (some can be repeated)

    # Grid for grid search
    param_dist = {
        'n_estimators': [250],  # [100, 150, 200, 250],
        'max_depth': [30],  # [None, 10, 20, 30, 40],
        'min_samples_split': [2],  # [2, 5, 10, 15],
        'min_samples_leaf': [1]  # [1, 2, 4, 6],
        # Include other hyperparameters as needed
    }

    # Create a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                 bootstrap=bootstrap, oob_score=oob_score, random_state=random_state,
                                 warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                 max_samples=max_samples)

    # Create Stratified K-fold cross validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    scoring = ('f1_weighted', 'recall_weighted', 'precision_weighted')
    # Evaluate SRF model
    scores = cross_validate(clf, over_X, over_y, scoring=scoring, cv=cv)
    # Get average evaluation metrics
    print('Mean f1: %.3f' % mean(scores['test_f1_weighted']))
    print('Mean recall: %.3f' % mean(scores['test_recall_weighted']))
    print('Mean precision: %.3f' % mean(scores['test_precision_weighted']))

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, scoring='f1_weighted', cv=cv,
                                       verbose=2)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_estimator = random_search.best_estimator_
    best_score = random_search.best_score_
    feature_importances = best_estimator.feature_importances_

    y_pred = random_search.predict(X_val)

    # Calculate validation metrics, take weighted average since we are considering multi classes
    f1_val = f1_score(y_val, y_pred, average='weighted')
    recall_val = recall_score(y_val, y_pred, average='weighted')
    precision_val = precision_score(y_val, y_pred, average='weighted')

    # Display metrics
    print("Best hyperparameters:", best_params)
    print("Best F1 (weighted) score:", best_score)
    print("F1 (weighted) on validation set:", f1_val)
    print("Recall (weighted) on validation set:", recall_val)
    print("Precision (weighted) on validation set:", precision_val)

    # Confusion matrix and classification report
    confusion = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)

    y_pred_test = random_search.predict(X_test)

    # Calculate validation metrics, take weighted average since we are considering multi classes
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')

    # Display metrics
    print("F1 (weighted) on test set:", f1_test)
    print("Recall (weighted) on test set:", recall_test)
    print("Precision (weighted) on test set:", precision_test)

    # Confusion matrix and classification report
    confusion = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=np.unique(y_test).astype(str))
    print("Classification Report:\n", report)

    # Convert the classification report to a DataFrame
    report_df = pd.read_fwf(StringIO(report), index_col=0)

    # Convert the DataFrame to an HTML table
    html_table = report_df.to_html(classes='table table-striped')

    # Save the HTML table to a file
    with open('/home/lucas/BBP/Code/Images/classification_report.html', 'w') as file:
        file.write(html_table)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    class_labels = np.unique(y_test)
    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.yticks(range(len(class_labels)), class_labels, rotation=45)
    plt.tight_layout()
    plt.savefig("/home/lucas/BBP/Code/Images/confusion_matrix.png")
    plt.show()

    # 1. ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(np.unique(y_test))
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_pred_encoded = label_encoder.transform(y_pred_test)

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test_encoded == i).astype(int), (y_pred_encoded == i).astype(int))
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = get_distinct_colors(n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{np.unique(y_test)[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve((y_test_encoded == i).astype(int),
                                                            (y_pred_encoded == i).astype(int))
        average_precision[i] = auc(recall[i], precision[i])

    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'{np.unique(y_test)[i]} (AP = {average_precision[i]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Sort the feature importances in descending order and get the corresponding feature names
    indices = feature_importances.argsort()[::-1]
    feature_names = X.columns

    # Visualize the top N most important features
    top_n = len(feature_names)  # You can change this number based on your needs
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(top_n), feature_importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("/home/lucas/BBP/Code/Images/feature_importance.png")
    plt.show()
