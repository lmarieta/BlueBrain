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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from MLPDropout import MLPDropout
from get_features import get_features
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import smote_variants as sv
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import to_categorical


def get_distinct_colors(num_colors):
    # Create a color map from the available colors
    colormap = plt.get_cmap("tab20")

    # Generate a list of distinct colors
    colors = [colormap(i) for i in range(num_colors)]

    return colors


def random_forest(X, y, random_state=42, parameter_scan=False):
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
    bootstrap = False  # Build trees with random sub-sample from dataset if True, otherwise use whole dataset
    oob_score = False  # If true estimate the performance of the model without the need for a separate validation set
    warm_start = False  # When set to True, reuse the solution of the previous call to fit and add more estimators
    class_weight = 'balanced'  # Trying balanced because of imbalanced dataset
    ccp_alpha = 0  # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost
    # complexity that is smaller than ccp_alpha will be chosen.
    max_samples = None  # None means each training set as the same size as original set (some can be repeated)

    if parameter_scan:
        # Grid for grid search
        param_dist = {
            'n_estimators': [5, 10, 100, 150],
            'max_depth': [6, 8, 10, 12, 14],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'ccp_alpha': [0, 0.5, 1]
            # Include other hyperparameters as needed
        }
    else:
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
    scores = cross_validate(clf, X, y, scoring=scoring, cv=cv)
    # Get average evaluation metrics
    print('Mean f1: %.3f' % mean(scores['test_f1_weighted']))
    print('Mean recall: %.3f' % mean(scores['test_recall_weighted']))
    print('Mean precision: %.3f' % mean(scores['test_precision_weighted']))

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, scoring='f1_weighted', cv=cv,
                                       random_state=random_state)

    return random_search


def logistic_regression():
    # Initialize the linear regression model
    model = LogisticRegression()
    # Define the hyperparameter search space
    param_dist = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
        'max_iter': [100, 200, 300, 400, 500],
        'class_weight': [None, 'balanced']
    }

    # Initialize RandomizedSearchCV
    model = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42,
                               n_jobs=-1)

    return model


def neural_network():
    # Initialize the neural network model
    model = MLPDropout()

    # Define the hyperparameter search space
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [5000, 10000, 20000],
        'early_stopping': [True, False],
        #  'dropout': [0.05, 0.1, 0.2]
    }

    # Initialize RandomizedSearchCV
    model = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)

    return model


def support_vector_machine(kernel='linear', C=1.0):
    svm_model = SVC(decision_function_shape='ovr')
    # Define the hyperparameter search space
    param_dist = {
        'C': np.logspace(-2, 2, 7),  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
        'degree': [2, 3, 4],  # Degree for poly kernel
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),  # Kernel coefficient for 'rbf' and 'poly'
        # 'class_weight': ['balanced']
    }

    svm_model = RandomizedSearchCV(svm_model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42,
                                   n_jobs=-1)
    return svm_model


def knearest_neighbours():
    knn = KNeighborsClassifier(n_neighbors=15)
    # Define the parameter grid
    param_dist = {
        'weights': ['distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['minkowski', 'manhattan', 'euclidean'],
    }

    # Initialize RandomizedSearchCV
    knn = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=5, random_state=42,
                             n_jobs=-1)
    return knn


def custom_nn(X, num_classes):
    # Define the neural network model
    model = Sequential()

    # Add an input layer with the appropriate input shape
    model.add(Dense(units=64, activation='relu', input_shape=(X.shape[1],)))

    # Add one or more hidden layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))

    # Add the output layer with the appropriate number of units (equal to the number of classes) and softmax activation
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def oversample(X, y, oversampler='SMOTE', random_state=42):
    if oversampler != 'SMOTE':
        # Use SMOTE to oversample the minority classes
        # Identify the two most frequent classes
        top_class = y.value_counts().idxmax()

        # Filter indices of the two most frequent class
        indices_not_top_class = y[y != top_class].index
        indices_top_class = y[y == top_class].index

        oversampler = sv.MulticlassOversampling(oversampler=oversampler, oversampler_params={'random_state': random_state})
        X_samp, y_samp = oversampler.sample(X.loc[indices_not_top_class], y.loc[indices_not_top_class])
        X_samp = pd.DataFrame(X_samp, columns=X.columns)
        y_samp = pd.Series(y_samp)
        X = pd.concat([X_samp, X.loc[indices_top_class]], ignore_index=True)
        y = pd.concat([y_samp, y.loc[indices_top_class]], ignore_index=True)
    else:
        oversampler = SMOTE(random_state=random_state)
        X, y = oversampler.fit_resample(X, y)
    return X, y


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    outliers_to_be_removed = '/home/lucas/BBP/Code/outliers.txt'
    threshold_detector = 'spikecount'
    protocols = ['IDRest', 'APWaveform']
    data = pd.DataFrame()
    read_json_files = True
    if read_json_files:
        for protocol in protocols:
            feature_names = get_features(protocol)
            feature_names = [item for item in feature_names if item != 'stim']
            feature_names = [item for item in feature_names if item != 'AP_width']

            data = pd.concat(
                [data, data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names, rm_in_cells=True,
                                          threshold_detector=threshold_detector, repetition_index=0,
                                          protocol_to_plot=protocol, cells_to_be_removed=outliers_to_be_removed)])
    else:
        data = pd.read_csv('/home/lucas/BBP/Code/aecode_data.csv', sep=' ')
        feature_names = get_features(protocols[0])
        feature_names = [item for item in feature_names if item != 'stim']
    data = data.reset_index(drop=True)
    original_indices = data.index

    parameter_scan = True
    interaction_terms = True
    model_type = 'custom_nn'  # pick from ['neural_network', 'SVM', 'logistic_regression', 'random_forest',
    # 'knn', 'custom_nn']
    oversampling = True
    random_state = 42

    # Split your data into training and validation sets
    X = data[feature_names]
    y = data['Group']
    y = pd.Series([label.split(' AP')[0] for label in y])

    # Split your data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # Oversample data because of imbalanced dataset
    if oversampling:
        X_train, y_train = oversample(X=X_train, y=y_train, oversampler='SMOTE')  # these are also good ADASYN, MSYN, SMOTE_Cosine, SMOTE_D, NT_SMOTE

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    short_labels = [label.split(' AP')[0] for label in y_test]
    class_labels = np.unique(short_labels)

    match model_type:
        case 'random_forest':
            model = random_forest(X, y, parameter_scan=parameter_scan)
        case 'logistic_regression':
            model = logistic_regression()
            if interaction_terms:
                # Use PolynomialFeatures to create interaction terms
                degree = 2  # You can adjust the degree as needed
                poly = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
                X_train = poly.fit_transform(X_train)
                X_test = poly.transform(X_test)
        case 'SVM':
            model = support_vector_machine()
        case 'neural_network':
            model = neural_network()
        case 'knn':
            model = knearest_neighbours()
        case 'custom_nn':
            model = custom_nn(X_train, len(class_labels))
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            y_train = to_categorical(y_train, num_classes=len(class_labels))

    clf = model.fit(X_train, y_train, epochs=200)

    if model_type == 'custom_nn':
        best_model = model
    else:
        best_model = model.best_estimator_

    if model_type == 'neural_network':
        # Access the training loss history
        training_loss = best_model.loss_curve_

        # Plot the convergence curve
        plt.plot(training_loss)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.show()

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    if model_type == 'custom_nn':
        y_pred_train_indices = np.argmax(y_pred_train, axis=1)
        y_pred_test_indices = np.argmax(y_pred_test, axis=1)
        y_pred_train = label_encoder.inverse_transform(y_pred_train_indices)
        y_pred_test = label_encoder.inverse_transform(y_pred_test_indices)
        y_train_indices = np.argmax(y_train, axis=1)
        y_train = label_encoder.inverse_transform(y_train_indices)
        y_test = label_encoder.inverse_transform(y_test)

    # Calculate validation metrics, take weighted average since we are considering multi classes
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    precision_train = precision_score(y_train, y_pred_train, average='weighted')

    print("F1 (weighted) on train set:", f1_train)
    print("Recall (weighted) on train set:", recall_train)
    print("Precision (weighted) on train set:", precision_train)

    # Confusion matrix and classification report
    confusion = confusion_matrix(y_train, y_pred_train)
    report = classification_report(y_train, y_pred_train)
    print("Train confusion matrix:\n", confusion)
    print("Train classification report:\n", report)

    # Confusion matrix and classification report
    confusion = confusion_matrix(y_test, y_pred_test)
    print("Test confusion matrix:\n", confusion)

    if parameter_scan and model_type != 'custom_nn':
        # retrieve metrics
        best_params = model.best_params_
        best_estimator = model.best_estimator_
        best_score = model.best_score_
        # feature_importances = best_estimator.feature_importances_

        # Display metrics
        print("Best hyperparameters:", best_params)
        print("Best F1 (weighted) score:", best_score)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix, ' + model_type)
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=90)
    plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)
    plt.tight_layout()
    plt.savefig("/home/lucas/BBP/Code/Images/confusion_matrix.png")
    plt.show()

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
    print("Test classification Report:\n", report)

    # Convert the classification report to a DataFrame
    report_df = pd.read_fwf(StringIO(report), index_col=0)

    # Convert the DataFrame to an HTML table
    html_table = report_df.to_html(classes='table table-striped')

    # Save the HTML table to a file
    with open('/home/lucas/BBP/Code/Images/classification_report.html', 'w') as file:
        file.write(html_table)

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
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{np.unique(y_test)[i]} (AP = {average_precision[i]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure()
    result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42)

    # Visualize feature importance
    plt.bar(range(X.shape[1]), result.importances_mean)
    plt.title("Permutation Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()
