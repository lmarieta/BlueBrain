from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from statistics import mean
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pandas import DataFrame

if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['AP_begin_voltage', 'AP_amplitude', 'AP_width']
    threshold_detector = 'spikecount'
    protocols = ['APWaveform']
    data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                                      threshold_detector=threshold_detector, protocols_to_plot=protocols)

    random_state = 0

    # Exclude IN cells and merge PC-L2 with PC-L5
    data = data[data['CellTypeGroup'] != 'IN']
    data['Group without cell type'] = data['Species'] + '_' + data['BrainArea']

    # Split your data into training and validation sets
    X = data[['AP_begin_voltage']]
    y = data['Group without cell type']

    # Use SMOTE to oversample the minority class
    oversample = SMOTE()
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
    param_grid = {
        'n_estimators': 100,  # [100, 150, 200],
        'max_depth': None,  # [None, 10, 20, 30],
        'min_samples_split': 2,  # [2, 5, 10],
        'min_samples_leaf': 1  #[1, 2, 4]
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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    scoring = ('f1_weighted', 'recall_weighted', 'precision_weighted')
    # Evaluate SRF model
    scores = cross_validate(clf, over_X, over_y, scoring=scoring, cv=cv)
    # Get average evaluation metrics
    print('Mean f1: %.3f' % mean(scores['test_f1_weighted']))
    print('Mean recall: %.3f' % mean(scores['test_recall_weighted']))
    print('Mean precision: %.3f' % mean(scores['test_precision_weighted']))

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1_weighted', cv=cv, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_

    y_pred = grid_search.predict(X_val)

    # Calculate validation metrics, take weighted average since we are considering multi classes
    f1_val = f1_score(y_val, y_pred, average='weighted')
    recall_val = recall_score(y_val,y_pred, average='weighted')
    precision_val = precision_score(y_val, y_pred, average='weighted')

    # Display metrics
    print("F1-score:", f1_val)
    print('Recall: ', recall_val)
    print('Precision: ', precision_val)

    print("F1-score:",best_score)

    # TODO test score
